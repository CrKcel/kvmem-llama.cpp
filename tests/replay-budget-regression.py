"""GPU integration regression for text-only query replay exceeding the KV budget.

Requires an MTP-capable model; starts private servers and stops only those servers.
Every SSE response, request and server log is retained in --out for diagnosis.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--mmproj', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--gpu', required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpu
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
    results = []
    for mtp, budget, mode in [(True, 1024, 'auto'), (False, 1024, 'auto'),
                              (True, 8192, 'auto'), (True, 1024, 'legacy')]:
        tag = f'{"mtp" if mtp else "plain"}-{budget}-{mode}'
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
                   '--kvmem-query-policy', 'user', '--kvmem-query-replay', mode,
                   '--reasoning-effort', 'none', '--temp', '0']
        (args.out / f'{tag}.argv.json').write_text(json.dumps(command, indent=2))
        log_path = args.out / f'{tag}.server.log'
        proc = None

        def request(name, msgs, reset=True, query=None, expected_path=None):
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
                  and 'READY' in content and finishes == ['stop']
                  and not any(s in trace for s in ['has no GPU slot', 'multimodal_error', 'multimodal_rollback']))
            if expected_path:
                ok = ok and any(f'path={expected_path} ' in line for line in decision)
            record = {'tag': tag, 'case': name, 'pass': ok, 'status': status,
                      'seconds': round(time.monotonic() - started, 2), 'content': content,
                      'finish': finishes, 'usage': usage, 'decision': decision}
            results.append(record)
            (args.out / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps(record), flush=True)
            if not ok:
                raise AssertionError(f'Failed {tag}/{name}; inspect {prefix}.sse.txt and server log')
            return content, usage

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
                answer, usage = request('long_tail', messages, expected_path=
                                        'query_replay_skipped' if budget == 1024 else 'all_resident')
                if budget == 1024 and mode == 'auto':
                    continuation = messages + [
                        {'role': 'assistant', 'content': answer, 'tool_calls': [
                            {'id': 'call_again', 'type': 'function', 'function':
                                {'name': 'read_records', 'arguments': '{}'}}]},
                        {'role': 'tool', 'tool_call_id': 'call_again', 'content': 'Final status remains READY.'}]
                    request('continuation', continuation, reset=False)
                    request('fresh_user', messages + [{'role': 'user', 'content':
                            'Now report the final status in one word.'}], expected_path='query_replay')
                    # 31 suffix blocks + one disjoint sink block exactly fill budget.
                    end = usage['prompt_tokens'] - int(mtp)
                    fit_begin = ((end + 31) // 32 - 31) * 32
                    request('exact_fit', messages, query=fit_begin, expected_path='query_replay')
                    request('one_row_over', messages, query=fit_begin - 1,
                            expected_path='query_replay_skipped')
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
