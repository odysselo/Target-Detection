# -*- coding: utf-8 -*-
"""
docs/md2docx.py
=========================================================================
把 Markdown 实验报告转换成排版好的 Word 文档

流程：
    1. 用 pandoc 导出默认 reference.docx
    2. 用 python-docx 修改样式（正文 宋体 10.5pt / 标题 黑体 / 代码 Consolas）
    3. pandoc 以该 reference 为模板转换，附带目录

用法： python docs/md2docx.py
=========================================================================
"""
import os
import shutil
import subprocess

# pandoc 可能不在 PATH 中（Windows 默认装在用户目录），这里做一次兜底查找
PANDOC = shutil.which("pandoc") or os.path.expanduser(
    r"~\AppData\Local\Pandoc\pandoc.exe")

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
SRC = os.path.join(DOCS, "实验报告.md")
REF = os.path.join(DOCS, "reference.docx")
DST = os.path.join(DOCS, "实验报告.docx")


def set_east_asian(run_or_style, latin: str, east_asian: str, size: float = None,
                   bold: bool = None, color: RGBColor = None):
    font = run_or_style.font
    font.name = latin
    if size is not None:
        font.size = Pt(size)
    if bold is not None:
        font.bold = bold
    if color is not None:
        font.color.rgb = color
    # 中文字体必须通过 rPr/rFonts 的 eastAsia 属性设置
    rpr = run_or_style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), east_asian)
    rfonts.set(qn("w:ascii"), latin)
    rfonts.set(qn("w:hAnsi"), latin)


def build_reference():
    subprocess.run([PANDOC, "--print-default-data-file", "reference.docx"],
                   stdout=open(REF, "wb"), check=True)

    doc = Document(REF)

    # 正文：宋体 10.5pt（五号），行距 1.5
    normal = doc.styles["Normal"]
    set_east_asian(normal, "Times New Roman", "宋体", 10.5)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(3)

    # 标题：黑体
    sizes = {"Heading 1": 18, "Heading 2": 15, "Heading 3": 13, "Heading 4": 11.5,
             "Heading 5": 11, "Heading 6": 10.5}
    for name, size in sizes.items():
        try:
            st = doc.styles[name]
        except KeyError:
            continue
        set_east_asian(st, "Times New Roman", "黑体", size, bold=True,
                       color=RGBColor(0x1F, 0x2A, 0x37))
        st.paragraph_format.space_before = Pt(10)
        st.paragraph_format.space_after = Pt(4)

    # 代码块
    for name in ("Source Code", "Verbatim Char"):
        try:
            st = doc.styles[name]
            set_east_asian(st, "Consolas", "Consolas", 8.5)
        except KeyError:
            pass

    # 表格文字稍小
    try:
        st = doc.styles["Table"]
        set_east_asian(st, "Times New Roman", "宋体", 9)
    except KeyError:
        pass

    doc.save(REF)
    print("reference 样式已生成:", REF)


def convert():
    cmd = [
        PANDOC, SRC, "-o", DST,
        "--reference-doc", REF,
        "--toc", "--toc-depth=3",
        "--resource-path", DOCS,
        "--highlight-style", "tango",
    ]
    subprocess.run(cmd, check=True, cwd=DOCS)
    print("已生成:", DST)


if __name__ == "__main__":
    build_reference()
    convert()
    size = os.path.getsize(DST) / 1024
    print(f"文件大小: {size:.0f} KB")
