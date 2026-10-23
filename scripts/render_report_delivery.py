"""Render the concise delivery report from Markdown to HTML and DOCX.

The input Markdown is deliberately small and uses a restricted subset:
ATX headings, paragraphs, bullet lists, GFM-style tables, fenced code,
blockquotes, bold text, inline code, and local PNG images.
"""

import argparse
import datetime
import html
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _inline(text):
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def markdown_to_html(markdown, css_path):
    lines = markdown.splitlines()
    body = []
    index = 0
    in_list = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            if in_list:
                body.append("</ul>")
                in_list = False
            index += 1
            continue
        if stripped.startswith("```"):
            if in_list:
                body.append("</ul>")
                in_list = False
            language = stripped[3:].strip()
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            body.append(
                f'<pre class="code {html.escape(language)}"><code>'
                + html.escape("\n".join(block))
                + "</code></pre>"
            )
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            if in_list:
                body.append("</ul>")
                in_list = False
            # Keep the image URI relative to the generated HTML so the delivery
            # package remains portable and print pages never expose host paths.
            source = image.group(2).replace("\\", "/")
            body.append(
                f'<figure><img src="{html.escape(source)}" alt="{html.escape(image.group(1))}">'
                f"<figcaption>{_inline(image.group(1))}</figcaption></figure>"
            )
            index += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            if in_list:
                body.append("</ul>")
                in_list = False
            level = len(heading.group(1))
            body.append(f"<h{level}>{_inline(heading.group(2))}</h{level}>")
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if re.fullmatch(r"\|?[\s:|-]+\|?", separator) and "-" in separator:
                if in_list:
                    body.append("</ul>")
                    in_list = False
                rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                    index += 1
                body.append('<div class="table-wrap"><table>')
                body.append("<thead><tr>"
                            + "".join(f"<th>{_inline(cell)}</th>" for cell in rows[0])
                            + "</tr></thead><tbody>")
                for row in rows[2:]:
                    body.append("<tr>" + "".join(f"<td>{_inline(cell)}</td>" for cell in row)
                                + "</tr>")
                body.append("</tbody></table></div>")
                continue
        if stripped.startswith("- "):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{_inline(stripped[2:])}</li>")
            index += 1
            continue
        if stripped.startswith(">"):
            if in_list:
                body.append("</ul>")
                in_list = False
            body.append(f"<blockquote>{_inline(stripped.lstrip('> '))}</blockquote>")
            index += 1
            continue
        if stripped in {"---", "***"}:
            body.append("<hr>")
            index += 1
            continue
        if in_list:
            body.append("</ul>")
            in_list = False
        paragraph = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (not candidate or candidate.startswith(("#", "- ", ">", "|", "```"))
                    or re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", candidate)):
                break
            paragraph.append(candidate)
            index += 1
        body.append("<p>" + _inline(" ".join(paragraph)) + "</p>")
    if in_list:
        body.append("</ul>")
    with open(css_path, encoding="utf-8") as handle:
        css = handle.read()
    return (
        "<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>RAG 课程实践报告（验收修订版）</title>"
        f"<style>{css}</style></head><body><main>{''.join(body)}</main></body></html>\n"
    )


def _set_east_asia(run, name):
    from docx.oxml.ns import qn
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)


def _add_runs(paragraph, text):
    from docx.shared import Pt
    parts = re.split(r"(`[^`]+`|\*\*[^*]+\*\*)", text)
    for part in parts:
        if not part:
            continue
        run = paragraph.add_run(part[1:-1] if part.startswith("`") else
                                part[2:-2] if part.startswith("**") else part)
        _set_east_asia(run, "Microsoft YaHei")
        run.font.size = Pt(9.5)
        if part.startswith("`"):
            run.font.name = "Consolas"
        elif part.startswith("**"):
            run.bold = True


def markdown_to_docx(markdown, output_path, image_root):
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Cm, Pt, RGBColor

    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(9.5)
    normal._element.rPr.rFonts.set(
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia",
        "Microsoft YaHei")
    for name, size in [("Title", 20), ("Heading 1", 14), ("Heading 2", 11.5),
                       ("Heading 3", 10.5)]:
        style = styles[name]
        style.font.name = "Microsoft YaHei"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(28, 55, 75)
        style._element.rPr.rFonts.set(
            "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}eastAsia",
            "Microsoft YaHei")

    lines = markdown.splitlines()
    index = 0
    in_list = False
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("```"):
            language = stripped[3:].strip()
            index += 1
            block = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            index += 1
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Cm(0.35)
            para.paragraph_format.space_after = Pt(4)
            run = para.add_run("\n".join(block))
            run.font.name = "Consolas"
            run.font.size = Pt(8)
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image:
            source = os.path.join(image_root, image.group(2))
            if os.path.isfile(source):
                doc.add_picture(source, width=Cm(15.7))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cap = doc.add_paragraph()
                cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                _add_runs(cap, image.group(1))
            index += 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            paragraph = doc.add_paragraph(style="Title" if level == 1 else
                                          "Heading 1" if level == 2 else
                                          "Heading 2" if level == 3 else "Heading 3")
            _add_runs(paragraph, heading.group(2))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if re.fullmatch(r"\|?[\s:|-]+\|?", separator) and "-" in separator:
                rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    rows.append([cell.strip() for cell in
                                 lines[index].strip().strip("|").split("|")])
                    index += 1
                width = max(len(row) for row in rows)
                table = doc.add_table(rows=len(rows), cols=width)
                table.style = "Table Grid"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                for row_index, row in enumerate(rows):
                    for col_index, cell in enumerate(row):
                        paragraph = table.cell(row_index, col_index).paragraphs[0]
                        _add_runs(paragraph, cell)
                        for run in paragraph.runs:
                            run.font.size = Pt(7.5)
                            if row_index == 0:
                                run.bold = True
                doc.add_paragraph().paragraph_format.space_after = Pt(0)
                continue
        if stripped.startswith("- "):
            paragraph = doc.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.space_after = Pt(1)
            _add_runs(paragraph, stripped[2:])
            index += 1
            continue
        if stripped.startswith(">"):
            paragraph = doc.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.35)
            paragraph.paragraph_format.space_after = Pt(3)
            _add_runs(paragraph, stripped.lstrip("> "))
            for run in paragraph.runs:
                run.italic = True
                run.font.color.rgb = RGBColor(75, 75, 75)
            index += 1
            continue
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(3)
        paragraph.paragraph_format.line_spacing = 1.12
        _add_runs(paragraph, stripped)
        index += 1
    doc.core_properties.title = "RAG 课程实践报告（验收修订版）"
    doc.core_properties.comments = (
        "Generated from 报告_精简.md. Some acceptance items remain human-required.")
    doc.core_properties.modified = datetime.datetime(2026, 10, 23)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    doc.save(output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", default=os.path.join(
        PROJECT_ROOT, "repair_outputs", "acceptance_20261023",
        "delivery", "报告_精简.md"))
    parser.add_argument("--html", default=os.path.join(
        PROJECT_ROOT, "repair_outputs", "acceptance_20261023",
        "delivery", "报告_精简.html"))
    parser.add_argument("--docx", default=os.path.join(
        PROJECT_ROOT, "repair_outputs", "acceptance_20261023",
        "delivery", "报告_精简.docx"))
    parser.add_argument("--css", default=os.path.join(
        PROJECT_ROOT, "repair_outputs", "acceptance_20261023",
        "delivery", "报告_打印样式.css"))
    args = parser.parse_args()
    with open(args.markdown, encoding="utf-8") as handle:
        markdown = handle.read()
    with open(args.html, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown_to_html(markdown, args.css))
    markdown_to_docx(markdown, args.docx, os.path.dirname(args.markdown))
    print(f"[report] html={os.path.abspath(args.html)}")
    print(f"[report] docx={os.path.abspath(args.docx)}")


if __name__ == "__main__":
    main()
