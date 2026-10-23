"""Independently validate a new submission archive before extracting it."""
import argparse
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', required=True, type=Path)
    parser.add_argument('--extract-to', required=True, type=Path)
    args = parser.parse_args()
    errors = []

    def check(ok, message):
        if not ok:
            errors.append(message)

    with zipfile.ZipFile(args.zip) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos if not i.is_dir()]
        check(len(names) == len(set(names)) == len({s.casefold() for s in names}), 'duplicate archive paths')
        for info in infos:
            p = PurePosixPath(info.filename)
            check(not p.is_absolute() and '..' not in p.parts and '\\' not in info.filename and ':' not in info.filename, 'unsafe archive path: ' + info.filename)
            check(not stat.S_ISLNK(info.external_attr >> 16), 'symlink: ' + info.filename)
            check(not any(s.lower() in {'.git', '.env', '__pycache__', '_cache', '.pytest_cache'} for s in p.parts), 'forbidden archive path: ' + info.filename)
            check(not info.filename.lower().endswith(('.pyc', '.pem', '.key')), 'secret or bytecode path: ' + info.filename)
        if errors:
            raise SystemExit(json.dumps(errors, ensure_ascii=False))
        check(archive.testzip() is None, 'ZIP CRC mismatch')
        manifest = json.loads(archive.read('delivery/manifest.json'))
        rows = manifest['files']
        by_path = {r['path']: r for r in rows}
        expected = set(names) - {'delivery/manifest.json', 'delivery/SHA256SUMS.txt'}
        check(len(rows) == len(by_path) == manifest['file_count'], 'manifest count or duplicate')
        check(set(by_path) == expected, 'manifest exact inventory mismatch')
        for path, row in by_path.items():
            data = archive.read(path)
            check(row['sha256'] == digest(data) and row['bytes'] == len(data), 'manifest mismatch: ' + path)
        checksum_rows = archive.read('delivery/SHA256SUMS.txt').decode('utf-8').splitlines()
        checksums = {}
        for line in checksum_rows:
            h, path = line.split('  ', 1)
            check(path not in checksums, 'duplicate checksum path: ' + path)
            checksums[path] = h
            check(path in names and digest(archive.read(path)) == h, 'checksum mismatch: ' + path)
        check(set(checksums) == set(names) - {'delivery/SHA256SUMS.txt'}, 'checksum exact inventory mismatch')
        formal = json.loads((ROOT / 'repair_outputs/acceptance_20261023/delivery/formal_results_snapshot.json').read_text(encoding='utf-8'))['files']
        formal_names = [n for n in names if n.startswith('results/') and n.count('/') == 1]
        check(len(formal_names) == 42 and {Path(n).name for n in formal_names} == set(formal), 'formal results inventory mismatch')
        for name, row in formal.items():
            check(digest(archive.read('results/' + name)) == row['sha256'], 'formal result changed: ' + name)
        corpus = [n for n in names if n.startswith('data/corpus/') and n.endswith('.docx')]
        check(len(corpus) == 43, 'corpus count must be 43')
        for n in corpus:
            check(digest(archive.read(n)) == digest((ROOT / n).read_bytes()), 'corpus changed: ' + n)
        qa_data = archive.read('data/eval/qa_set.jsonl')
        check(digest(qa_data) == 'bfcbefc1ba2519c9728451559d30fd55f05151f903a927d836011ba07ae7b969', 'QA hash mismatch')
        check(len([s for s in qa_data.decode('utf-8-sig').splitlines() if s.strip()]) == 30, 'QA count mismatch')
        required = ['scripts/verify_acceptance_bundle.py', 'scripts/main_acceptance_audit.py', 'scripts/main_final_acceptance_audit.py', 'results/acceptance_20261023/t3/control_rrf/control20/control_answers.csv', 'results/acceptance_20261023/t3/control_rrf/control20/control_answers.jsonl', 'results/acceptance_20261023/t3/control_rrf/control20/meta.json', 'results/acceptance_20261023/t3/control20_vs30.csv', 'results/acceptance_20261023/t3/control20_vs30_meta.json', 'results/acceptance_20261023/t7/judge_checkpoint.jsonl', 'results/acceptance_20261023/demo/http_e2e_results.json']
        for n in required:
            check(n in names, 'missing required artifact: ' + n)
        pdf_data = archive.read('\u62a5\u544a.pdf')
        check(pdf_data == archive.read('delivery/\u62a5\u544a_\u7cbe\u7b80.pdf'), 'root PDF is not the reviewed compact PDF')
        import fitz
        doc = fitz.open(stream=pdf_data, filetype='pdf')
        page_text = [p.get_text() for p in doc]
        check(1 <= len(doc) <= 8 and all(t.strip() for t in page_text), 'PDF page count or blank page')
        check(not any(re.search(r'file:///|C:\\Users\\', t) for t in page_text), 'host path leaked in report')
        pages = len(doc)
        doc.close()
        image_count = len(re.findall(r'!\[[^\]]*\]\([^\)]+\)', archive.read('delivery/\u62a5\u544a_\u7cbe\u7b80.md').decode('utf-8')))
        check(image_count >= 6, 'report missing required figures')
        report = {'status': 'FAIL' if errors else 'PASS', 'archive': str(args.zip.resolve()), 'archive_sha256': digest(args.zip.read_bytes()), 'files': len(names), 'corpus': len(corpus), 'formal_results': len(formal_names), 'manifest_entries': len(rows), 'checksum_entries': len(checksums), 'pdf_pages': pages, 'report_images': image_count, 'errors': errors}
        output = ROOT / 'repair_outputs/acceptance_20261023/main_zip_audit.json'
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if errors:
            return 1
        # Never merge into a prior extraction or a working project.
        args.extract_to.mkdir(parents=True, exist_ok=False)
        archive.extractall(args.extract_to)
        print('Extracted into new directory: ' + str(args.extract_to.resolve()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
