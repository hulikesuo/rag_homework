"""纯文本句子切分工具，供各压缩入口共享。

只依赖正则，不产生缓存文件。导入本模块不会加载模型、配置或生成器，
因此可被默认压缩和 InformationCompressor 等路径安全复用。
"""

import re

# Split Chinese/English sentence punctuation and line breaks, keeping punctuation.
# A decimal point such as the one in "3.14元" is not a sentence boundary.
# A closing bracket/quote right after terminal punctuation is kept WITH that
# sentence and the boundary is placed after it, so an English "Apple price."
# followed by more text stays whole and Chinese 他说"价格已公布。" must not split
# before the closing quote. A terminator not followed by a closing bracket splits
# immediately. These lookbehinds use only fixed-width alternatives so the pattern
# is a valid Python re (no variable-width lookbehind).
_CLOSING = "\"'”’　）)\\]」』"
_BRKT = "[" + _CLOSING + "]"
_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[。！？；!?;])(?!" + _BRKT + ")"           # terminator, no trailing bracket -> split
    r"|(?<=[。！？；!?;]" + _BRKT + ")"              # terminator + bracket -> split after both
    r"|(?<=\.)(?!" + _BRKT + r")(?=\s|$)"           # ASCII period -> space/EOL (not a decimal)
    r"|(?<=\." + _BRKT + r")(?=\s|$)"               # ASCII period + bracket -> split after both
    r"|[\r\n]+")


def split_sentences(text):
    """Split text into [] of trimmed sentences using fixed-width boundaries.

    Returns the same tokens as the boundary pattern produces: empty fragments
    are preserved so callers can decide how to filter them.
    """
    return _SENTENCE_BOUNDARY.split(text)
