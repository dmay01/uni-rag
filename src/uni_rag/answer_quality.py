"""Streaming thought-block suppression and transparent output-quality flags."""

import re
from collections import Counter


class VisibleText:
    """Keep incomplete tags buffered so neither split tags nor thought bodies leak."""

    def __init__(self):
        self.buffer = ""
        self.hidden = False
        self.think_tags_seen = False

    def feed(self, text: str, final: bool = False) -> str:
        self.buffer += text
        output = []
        while self.buffer:
            match = re.search(r"</?think\s*>", self.buffer, re.I)
            if match:
                if not self.hidden:
                    output.append(self.buffer[: match.start()])
                self.hidden = not match.group().startswith("</")
                self.think_tags_seen = True
                self.buffer = self.buffer[match.end() :]
                continue
            # A partial tag can span any number of streaming events.
            tail = self.buffer.rfind("<")
            hold = tail >= 0 and any(
                tag.startswith(self.buffer[tail:].lower()) for tag in ("<think>", "</think>")
            )
            if hold and not final:
                if not self.hidden:
                    output.append(self.buffer[:tail])
                self.buffer = self.buffer[tail:]
                break
            if not self.hidden:
                output.append(self.buffer)
            self.buffer = ""
        return "".join(output)


def quality(text: str) -> dict:
    words = re.findall(r"\w+", text.casefold())
    phrases = Counter(tuple(words[i : i + 12]) for i in range(max(0, len(words) - 11)))
    return {
        "answer_words": len(words),
        "repetition_flag": max(phrases.values(), default=0) >= 4,
        "max_repeated_12gram": max(phrases.values(), default=0),
        "visible_think_tag": bool(re.search(r"</?think\b", text, re.I)),
    }
