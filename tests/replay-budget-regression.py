"""GPU integration regression for query replay exceeding the KV budget.

Requires an MTP-capable model; starts private servers and stops only those servers.
Every SSE response, request and server log is retained in --out for diagnosis.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--mmproj', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--with-images', action='store_true', help='Exercise image groups and text together')
    parser.add_argument('--query-policy', choices=['user', 'legacy'], default='user')
    parser.add_argument('--configuration', action='append', help='Only run named configurations, e.g. mtp-1024-auto')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["KVMEM_TRACE"] = "1"
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['PATH'] = str(args.exe.parent) + os.pathsep + env['PATH']
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    messages = [
        {'role': 'system', 'content': 'Read the tool output and answer briefly.'},
        {'role': 'user', 'content': 'Inspect the records. What is the final status?'},
        {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call_records', 'type': 'function', 'function': {'name': 'read_records', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'call_records', 'content':
            'record value = green; verification = complete.\n' * 300 + '\nFinal status: READY.'},
    ]
    if args.with_images:
        # A deterministic red PNG fixture; no external image dependency.
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 224, 224, 8, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress((b'\0' + b'\xff\0\0' * 224) * 224)) + chunk(b'IEND', b''))
        (args.out / 'red.png').write_bytes(png)
        image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(png).decode()}}
        messages[0]['content'] += ' Context notes. ' * 80
        messages[1]['content'] = [image, {'type': 'text', 'text':
            'Inspect the image and records. Report the image color and final status briefly.'}]
    results = []
    configurations = [(True, 1024, 'auto'), (False, 1024, 'auto'),
                      (True, 8192, 'auto'), (True, 1024, 'legacy')]
    if args.with_images:
        configurations.append((True, 64, 'auto'))
    available = {f'{"mtp" if mtp else "plain"}-{budget}-{mode}' for mtp, budget, mode in configurations}
    if args.configuration and set(args.configuration) - available:
        parser.error('Unknown configuration for this mode; choose from ' + ', '.join(sorted(available)))
    for mtp, budget, mode in configurations:
        tag = f'{"mtp" if mtp else "plain"}-{budget}-{mode}'
        if args.configuration and tag not in args.configuration:
            continue
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        command = [str(args.exe), '-m', str(args.model), '--mmproj', str(args.mmproj),
                   '--no-mmproj-offload', '--host', '127.0.0.1', '--port', str(port),
                   '-c', '32768', '-n', '64', '-ngl', '99', '-b', '512', '--no-ui',
                   '--kvmem', '--kvmem-budget', str(budget), '--kvmem-gen-reserve', '2048',
                   '--kvmem-block-tokens', '32', '--kv-dtype', 'q8_0',
                   '--spec-type', 'draft-mtp' if mtp else 'none', '--spec-draft-n-max', '3',
                   '--kvmem-query-policy', args.query_policy, '--kvmem-query-replay', mode,
                   '--reasoning-effort', 'none', '--temp', '0']
        if args.with_images:
            command += ['--image-max-tokens', '256']
        (args.out / f'{tag}.argv.json').write_text(json.dumps(command, indent=2))
        log_path = args.out / f'{tag}.server.log'
        proc = None

        def request(name, msgs, reset=True, query=None, expected_path=None, expected_color=False,
                    expected_ready=True, expected_error=None):
            offset = log_path.stat().st_size
            payload = {'messages': msgs, 'max_tokens': 64, 'stream': True,
                       'cache_reset': reset, 'temperature': 0, 'reasoning_effort': 'none'}
            if query is not None:
                payload['kvmem'] = {'query_begin': query, 'query_end': query + 11}
            prefix = args.out / f'{tag}-{name}'
            raw = json.dumps(payload).encode()
            prefix.with_suffix('.request.json').write_bytes(raw)
            req = urllib.request.Request(base + '/v1/chat/completions', data=raw,
                                         headers={'Content-Type': 'application/json'})
            started = time.monotonic()
            try:
                with opener.open(req, timeout=240) as response:
                    status, body = response.status, response.read().decode()
            except urllib.error.HTTPError as error:
                status, body = error.code, error.read().decode()
            prefix.with_suffix('.sse.txt').write_text(body, encoding='utf-8')
            if expected_error:
                ok = status == 400 and expected_error in json.loads(body).get('error', '')
                record = {'tag': tag, 'case': name, 'pass': ok, 'status': status, 'error': body[:500]}
                results.append(record)
                (args.out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
                print(json.dumps(record), flush=True)
                assert ok, record
                return
            events = [json.loads(line[6:]) for line in body.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            content = ''.join(choice.get('delta', {}).get('content') or ''
                              for event in events for choice in event.get('choices', []))
            finishes = [choice.get('finish_reason') for event in events
                        for choice in event.get('choices', []) if choice.get('finish_reason')]
            usage = next((event['usage'] for event in events if 'usage' in event), {})
            with log_path.open('rb') as log:
                log.seek(offset)
                trace = log.read().decode(errors='replace')
            decision = re.findall(r'KVMEM_PREFILL_DECISION [^\r\n]+', trace)
            ok = (status == 200 and 'data: [DONE]' in body and not any('error' in e for e in events)
                  and (not expected_ready or 'READY' in content) and finishes == ['stop']
                  and not any(s in trace for s in ['has no GPU slot', 'multimodal_error', 'multimodal_rollback']))
            if expected_path:
                ok = ok and any(f'path={expected_path} ' in line for line in decision)
            if expected_color:
                ok = ok and 'red' in content.lower()
            record = {'tag': tag, 'case': name, 'pass': ok, 'status': status,
                      'seconds': round(time.monotonic() - started, 2), 'content': content,
                      'finish': finishes, 'usage': usage, 'decision': decision}
            results.append(record)
            (args.out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps(record), flush=True)
            if not ok:
                raise AssertionError(f'Failed {tag}/{name}; inspect {prefix}.sse.txt and server log')
            return content, usage, trace

        try:
            with log_path.open('wb') as log:
                proc = subprocess.Popen(command, env=env, stdout=log, stderr=log,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                print(f'Start {tag} pid={proc.pid}', flush=True)
                deadline = time.monotonic() + 240
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'Server exited {proc.returncode}: {log_path}')
                    try:
                        with opener.open(base + '/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError('Server startup')
                    time.sleep(1)
                if budget == 64:
                    request('oversized_image_group', messages, expected_error='image group exceeds KV budget')
                    continue
                # Legacy multimodal policy probes only the default 512-token
                # suffix. Explicit spans below still exercise its overflow path.
                low_budget_path = ('query_replay' if args.with_images and args.query_policy == 'legacy'
                                   else 'query_replay_skipped')
                answer, usage, trace = request('long_tail', messages, expected_path=
                                        low_budget_path if budget == 1024 else 'all_resident',
                                        expected_color=args.with_images)
                if budget == 1024 and mode == 'auto':
                    continuation = messages + [
                        {'role': 'assistant', 'content': answer, 'tool_calls': [
                            {'id': 'call_again', 'type': 'function', 'function':
                                {'name': 'read_records', 'arguments': '{}'}}]},
                        {'role': 'tool', 'tool_call_id': 'call_again', 'content': 'Final status remains READY.'}]
                    request('continuation', continuation, reset=False)
                    request('fresh_user', messages + [{'role': 'user', 'content':
                            'Now report the final status in one word.'}], expected_path='query_replay')
                    # Sink and image groups plus the disjoint suffix exactly fill budget.
                    end = usage['prompt_tokens'] - int(mtp)
                    hard = {0}
                    if args.with_images:
                        media = re.findall(r'multimodal_decode [^\n]*rows=\[(\d+),(\d+)\) [^\n]*image=1 replay=0', trace)
                        assert media, 'No visual rows were evaluated'
                        for begin, finish in media:
                            hard.update(range(max(0, int(begin) - 1) // 32, (int(finish) + 1 + 31) // 32))
                    fit_begin = ((end + 31) // 32 - (budget // 32 - len(hard))) * 32
                    assert fit_begin // 32 > max(hard)
                    request('exact_fit', messages, query=fit_begin, expected_path='query_replay')
                    request('one_row_over', messages, query=fit_begin - 1,
                            expected_path='query_replay_skipped')
                    if args.with_images:
                        request('multiple_images', [messages[0], {'role': 'user', 'content':
                            [image, image, {'type': 'text', 'text': 'What color are both images? Answer only the color.'}]},
                            *messages[2:]], expected_path=low_budget_path,
                            expected_color=True, expected_ready=False)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    print(f'PASS: {len(results)} requests; artifacts at {args.out}', flush=True)


if __name__ == '__main__':
    main()
