import re, sys, html, pathlib
O = pathlib.Path("theory_txt"); O.mkdir(exist_ok=True)
for f in sys.argv[1:]:
    t = open(f"docs/{f}.html", encoding="utf-8").read()
    body = re.search(r"<main>(.*)</main>", t, re.S).group(1)
    body = re.sub(r"<script.*?</script>|<svg.*?</svg>|<style.*?</style>|<pre.*?</pre>", "", body, flags=re.S)
    body = re.sub(r"</(p|li|tr|h[1-6]|div|blockquote|table)>", "\n\n", body)
    body = html.unescape(re.sub(r"<[^>]+>", "", body))
    body = re.sub(r"[ \t]+", " ", body); body = re.sub(r"\n[ \t]+", "\n", body); body = re.sub(r"\n{3,}", "\n\n", body)
    (O / f"{f}.txt").write_text(body)
