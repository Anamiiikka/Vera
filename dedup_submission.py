"""Deduplicate submission.jsonl — keep last occurrence of each test_id."""
import json

lines = [json.loads(l) for l in open('submission.jsonl', encoding='utf-8') if l.strip()]

seen = {}
for line in lines:
    seen[line['test_id']] = line

ordered = sorted(seen.values(), key=lambda x: int(x['test_id'][1:]))
print(f'Deduplicated to {len(ordered)} lines')
for l in ordered:
    flag = 'PLACEHOLDER' if 'Placeholder' in l.get('rationale','') else 'OK'
    body_preview = l['body'][:70]
    print(f"  {l['test_id']}: {flag} | {body_preview}")

with open('submission.jsonl', 'w', encoding='utf-8') as f:
    for l in ordered:
        f.write(json.dumps(l, ensure_ascii=False) + '\n')
print('Done.')
