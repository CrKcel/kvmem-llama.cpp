#!/usr/bin/env python3
"""Real IQ3/projector regression on the 5060 Ti. Starts only its own test server."""
import argparse
import base64
import ctypes
import csv
import json
import http.server
import threading
import os
from pathlib import Path
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request
import zlib

from mtp_kv_ab import Sampler, stop_server

ROOT = Path(__file__).resolve().parents[1]
GPU = 'GPU-5847813c-9e6e-bb43-cc5e-621aac091b6c'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fixture(size=896, changed=False):
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            a, b = x / size, y / size
            color = (250, 250, 250)
            if .08 < a < .42 and .15 < b < .55:
                color = (240, 190, 10) if changed else (225, 20, 25)
            if (a - .73)**2 + (b - .35)**2 < .17**2:
                color = (20, 65, 230)
            if .65 < b < .9 and abs(a - .5) < (b - .65)*.7:
                color = (20, 170, 55)
            row.extend(color)
        rows.append(row)

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)

    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b''.join(rows))) + chunk(b'IEND', b''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', choices=['gpu', 'cpu'], default='gpu')
    ap.add_argument('--spec', choices=['none', 'draft-mtp'], default='draft-mtp')
    ap.add_argument('--kv', default='q8_0')
    ap.add_argument('--draft-kv', default=None)
    ap.add_argument('--budget', type=int, default=4096)
    ap.add_argument('--reserve', type=int, default=2048)
    ap.add_argument('--ctx', type=int, default=16384)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--image-max-tokens', type=int, default=1024)
    ap.add_argument('--long-words', type=int, default=0)
    ap.add_argument('--port', type=int, default=18201)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--keep-server', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--performance', action='store_true')
    ap.add_argument('--no-kvmem', action='store_true')
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--expect-capacity-error', action='store_true')
    args = ap.parse_args()
    folder = ROOT / args.folder
    folder.mkdir(parents=True, exist_ok=True)
    print('ARTIFACTS', folder, flush=True)
    image = fixture()
    (folder / 'shapes.png').write_bytes(image)
    encoded = base64.b64encode(image).decode()
    part = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + encoded}}
    changed = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(fixture(changed=True)).decode()}}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               LD_LIBRARY_PATH=str(ROOT / 'build/bin') + ':/home/leye/kvmem_qw3/.cu13-env/lib',
               NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
    if args.trace:
        env['KVMEM_TRACE'] = '1'
    cmd = [str(ROOT / 'build/bin/llama-kvmem-server'),
           '-m', str(ROOT / 'models/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf'),
           '--mmproj', str(ROOT / 'models/unsloth/Qwen3.8-27B-GGUF/mmproj-Q8_0.gguf'),
           '--mmproj-offload' if args.device == 'gpu' else '--no-mmproj-offload',
           '--image-max-tokens', str(args.image_max_tokens), '--host', '127.0.0.1', '--port', str(args.port),
           '-c', str(args.ctx), '-b', str(args.batch), '-ngl', '99', '--kvmem', '--kvmem-method', 'retrieval',
           '--kvmem-budget', str(args.budget), '--kvmem-gen-reserve', str(args.reserve),
           '--kvmem-block-tokens', '128', '--kv-dtype', args.kv,
           '--spec-type', args.spec, '--spec-draft-n-max', '2', '--enable-thinking', '--reasoning-budget', '0']
    if args.draft_kv:
        cmd += ['--spec-kv-dtype', args.draft_kv]
    if args.no_kvmem:
        cmd += ['--no-kvmem']
    (folder / 'argv.json').write_text(json.dumps(cmd, indent=2))
    nvml = ctypes.CDLL('libnvidia-ml.so.1')
    assert nvml.nvmlInit_v2() == 0
    device = ctypes.c_void_p()
    assert nvml.nvmlDeviceGetHandleByUUID(GPU.encode(), ctypes.byref(device)) == 0
    sampler = Sampler(nvml, device, folder)
    with (folder / 'server.stderr.log').open('w') as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=fh)
    (folder / 'pid').write_text(str(proc.pid))
    sampler.start()
    rss_stop = threading.Event()
    rss_samples = []
    def sample_rss():
        start = time.monotonic()
        with (folder / 'rss.csv').open('w') as output:
            writer = csv.writer(output)
            writer.writerow(['elapsed_s', 'phase', 'rss_mib'])
            while not rss_stop.is_set():
                try:
                    status = Path(f'/proc/{proc.pid}/status').read_text()
                    match = re.search(r'^VmRSS:\s+(\d+)', status, re.M)
                    if match:
                        value = int(match[1]) / 1024
                        rss_samples.append(value)
                        writer.writerow([time.monotonic() - start, sampler.phase, value])
                        output.flush()
                except FileNotFoundError:
                    break
                rss_stop.wait(.2)
    rss_thread = threading.Thread(target=sample_rss, daemon=True)
    rss_thread.start()
    base = f'http://127.0.0.1:{args.port}'
    results = []

    def post(label, messages, expected=None, extra=None, status=200):
        payload = {'messages': messages, 'max_tokens': 128, 'temperature': 0, 'seed': 42, 'enable_thinking': False}
        payload.update(extra or {})
        (folder / (label + '.request.json')).write_text(json.dumps(payload, ensure_ascii=False))
        sampler.phase = label
        log_start = (folder / 'server.stderr.log').stat().st_size
        start = time.monotonic()
        request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
        try:
            with OPENER.open(request, timeout=1800) as response:
                code, raw = response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            code, raw = exc.code, exc.read().decode()
        (folder / (label + '.response.txt')).write_text(raw)
        assert code == status, (label, code, raw)
        if payload.get('stream') and code == 200:
            chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: {')]
            assert not any('error' in chunk for chunk in chunks), (label, raw)
            assert 'data: [DONE]' in raw, (label, raw)
            text = ''.join(choice.get('delta', {}).get('content', '')
                           for chunk in chunks for choice in chunk.get('choices', []))
            for word in expected or []:
                assert word in text.lower(), (label, word, text)
        data = json.loads(raw) if not payload.get('stream') else None
        if data and code == 200:
            message = data['choices'][0]['message']
            text = message.get('content') or ''
            for word in expected or []:
                assert word in text.lower(), (label, word, text)
        else:
            message = None
        tail = (folder / 'server.stderr.log').read_bytes()[log_start:].decode(errors='replace')
        (folder / (label + '.trace.log')).write_text(tail)
        result = {'trace': re.findall(r'KVMEM_TRACE multimodal_prefill (.*)', tail), 'label': label, 'status': code, 'elapsed_s': time.monotonic() - start, 'response': data}
        results.append(result)
        (folder / 'requests.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print('PASS', label, round(result['elapsed_s'], 2), str(message)[:180], flush=True)
        return message, data, raw

    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError((folder / 'server.stderr.log').read_text()[-3000:])
            try:
                with OPENER.open(base + '/health', timeout=1):
                    break
            except Exception:
                time.sleep(.5)
        else:
            raise RuntimeError('server startup timed out')
        history = []
        if args.long_words:
            history = [{'role': 'user', 'content': 'Remember this background and respond OK.\n' + ' apple' * args.long_words}]
            answer, _, _ = post('long-text', history)
            history.append(answer)
        history.append({'role': 'user', 'content': [part, {'type': 'text', 'text': 'Name the color and shape of each of the three objects. Be concise.'}]})
        if args.expect_capacity_error:
            post('capacity-error', history, status=400)
            post('after-capacity-error', [{'role': 'user', 'content': 'What is 2 + 2? Reply with one digit.'}], ['4'])
            return
        answer, data, _ = post('image', history, ['red', 'blue', 'green'])
        if args.long_words:
            assert data['usage']['prompt_cache_hit_tokens'] > 50000, data['usage']
        if args.performance and args.quick:
            perf_history = history + [answer, {'role': 'user', 'content':
                'Write a self-contained HTML page that draws the three colored shapes in the image using SVG. '
                'Include accessible labels and a button that changes the square color. Return the full code.'}]
            post('code-thinking', perf_history, extra={'temperature': 1.0, 'top_p': .95, 'top_k': 20,
                 'min_p': 0, 'presence_penalty': 0, 'frequency_penalty': 0, 'repetition_penalty': 1,
                 'enable_thinking': True, 'reasoning_budget_tokens': 128, 'max_tokens': 512})
        if args.quick:
            return
        history.append(answer)
        history.append({'role': 'user', 'content': 'What color is the circle? Reply with one word.'})
        _, follow, _ = post('followup', history, ['blue'])
        assert follow['usage']['prompt_cache_hit_tokens'] > 0, follow['usage']
        assert 'new_image_rows=0' in results[-1]['trace'][-1] and 'vision_encode_calls=0' in results[-1]['trace'][-1]
        post('stream', history, ['blue'], extra={'stream': True})
        bad = history + [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,bm90YW5pbWFnZQ=='}}]}]
        post('invalid-image', bad, status=400)
        post('after-error', history, ['blue'])
        # The URL is transport only; identical bytes reuse the same native image ID.
        class ImageHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Type', 'image/png')
                self.send_header('Content-Length', str(len(image)))
                self.end_headers()
                self.wfile.write(image)
            def log_message(self, *unused):
                pass
        image_server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), ImageHandler)
        thread = threading.Thread(target=image_server.serve_forever, daemon=True)
        thread.start()
        try:
            url_history = json.loads(json.dumps(history))
            image_index = next(i for i, m in enumerate(url_history) if isinstance(m.get('content'), list))
            url_history[image_index]['content'][0]['image_url']['url'] = f'http://127.0.0.1:{image_server.server_port}/shapes.png'
            post('http-image', url_history, ['blue'])
            assert 'vision_encode_calls=0' in results[-1]['trace'][-1]
        finally:
            image_server.shutdown()
            image_server.server_close()
            thread.join()
        # Cancel after SSE headers/role, while a new suffix is still being processed.
        cancel = {'messages': history + [{'role': 'user', 'content': 'Read this and say OK.' + ' apple' * 2000}],
                  'max_tokens': 128, 'stream': True, 'temperature': 0, 'enable_thinking': False}
        request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(cancel).encode(),
                                         headers={'Content-Type': 'application/json'})
        with OPENER.open(request, timeout=120) as response:
            response.readline()
        time.sleep(1)
        post('after-cancel', history, ['blue'])
        cancel_decode = {'messages': history + [{'role': 'user', 'content': 'List the integers from 1 to 1000, one per line.'}],
                         'max_tokens': 512, 'stream': True, 'temperature': 0, 'enable_thinking': False}
        request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(cancel_decode).encode(),
                                         headers={'Content-Type': 'application/json'})
        with OPENER.open(request, timeout=120) as response:
            for line in response:
                if not line.startswith(b'data: {'):
                    continue
                chunk = json.loads(line[6:])
                if any(c.get('delta', {}).get('content') for c in chunk.get('choices', [])):
                    break
        time.sleep(.5)
        post('after-decode-cancel', history, ['blue'])
        second = history + [{'role': 'assistant', 'content': 'Blue'},
                            {'role': 'user', 'content': [changed, {'type': 'text', 'text': 'What color is the square in this NEW image? One word.'}]}]
        post('second-image', second, ['yellow'])
        assert 'vision_encode_calls=1' in results[-1]['trace'][-1]
        # Same-sized image with different bytes must invalidate the old visual chunk.
        changed_history = history[:-2]
        changed_history[-1] = {'role': 'user', 'content': [changed, {'type': 'text', 'text': 'Name the color and shape of each object.'}]}
        post('changed-image', changed_history, ['yellow', 'blue', 'green'])
        tools = [{'type': 'function', 'function': {'name': 'record_shape', 'description': 'Record the circle color.',
                 'parameters': {'type': 'object', 'properties': {'color': {'type': 'string'}}, 'required': ['color']}}}]
        tool_history = [{'role': 'user', 'content': [part, {'type': 'text', 'text': 'Use record_shape to record the color of the circle in this image.'}]}]
        message, _, _ = post('image-tool', tool_history, extra={'tools': tools, 'tool_choice': 'required'})
        call = message['tool_calls'][0]
        assert call['function']['name'] == 'record_shape', message
        assert 'blue' in call['function']['arguments'].lower(), message
        tool_history += [message, {'role': 'tool', 'tool_call_id': call['id'], 'content': 'Recorded blue successfully.'}]
        post('tool-continuation', tool_history, extra={'tools': tools, 'tool_choice': 'none'})
        edited_tools = json.loads(json.dumps(tools))
        edited_tools[0]['function']['description'] = 'Store the color of the circle for later use.'
        _, edited, _ = post('history-edit', tool_history, extra={'tools': edited_tools, 'tool_choice': 'none'})
        assert edited['usage']['prompt_cache_hit_tokens'] == 0, edited['usage']
        _, reused, _ = post('after-history-edit', tool_history, extra={'tools': edited_tools, 'tool_choice': 'none'})
        assert reused['usage']['prompt_cache_hit_tokens'] > 0, reused['usage']
        _, reset, _ = post('explicit-cache-reset', tool_history,
                         extra={'tools': edited_tools, 'tool_choice': 'none', 'cache_reset': True})
        assert reset['usage']['prompt_cache_hit_tokens'] == 0, reset['usage']

        # Independent sessions can share a long system prefix. A title request can
        # also interrupt the single slot; neither case requires a custom reset field.
        system_prefix = 'You are a helpful assistant. Read each request carefully and answer concisely. '
        system = {'role': 'system', 'content': system_prefix + 'Use only the notes in the current conversation.'}
        session_a = [system,
                     {'role': 'user', 'content': 'Remember these notes. ' + ' apple' * 1200 + '\nThe secret color is ORANGE.'},
                     {'role': 'user', 'content': 'What is the secret color? Reply with one word.'}]
        post('cache-session-a', session_a, ['orange'])
        session_b = [system, {'role': 'user', 'content': 'What is 2 + 2? Reply with one digit.'}]
        answer, data, _ = post('cache-session-b', session_b, ['4'])
        assert data['usage']['prompt_cache_hit_tokens'] == 0, data['usage']
        reset_trace = (folder / 'cache-session-b.trace.log').read_text()
        miss = re.search(r'reason=no_recurrent_checkpoint lcp=(\d+)', reset_trace)
        assert miss and int(miss[1]) > 4, reset_trace
        assert 'cached_tail_rows=0' in results[-1]['trace'][-1]
        session_b += [answer, {'role': 'user', 'content': 'Repeat that result. Reply with one digit.'}]
        _, data, _ = post('cache-session-b-continue', session_b, ['4'])
        assert data['usage']['prompt_cache_hit_tokens'] > 0, data['usage']
        title = [{'role': 'system', 'content': system_prefix + 'Create a short session title. Return only the title.'},
                 {'role': 'user', 'content': 'We are discussing arithmetic.'}]
        post('cache-title-stream', title, extra={'stream': True})
        assert 'reason=no_recurrent_checkpoint' in (folder / 'cache-title-stream.trace.log').read_text()
        post('cache-session-a-return', session_a, ['orange'], extra={'stream': True})
        assert 'reason=no_recurrent_checkpoint' in (folder / 'cache-session-a-return.trace.log').read_text()

    finally:
        rss_stop.set()
        rss_thread.join()
        stats = sampler.finish()
        stats['peak_process_rss_mib'] = max(rss_samples, default=0)
        stats.update(options=vars(args), requests=results)
        (folder / 'summary.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
        print('PEAK_MIB', stats['peak_vram_mib'], flush=True)
        if not args.keep_server:
            stop_server(proc)
        nvml.nvmlShutdown()


if __name__ == '__main__':
    main()
