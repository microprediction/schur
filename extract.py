import re, sys, pathlib
D = pathlib.Path(sys.argv[1]); O = pathlib.Path(sys.argv[2]); O.mkdir(exist_ok=True)
for f in sorted(D.glob("*.html")):
    if f.name not in sys.argv[3:]: continue
    h = f.read_text()
    h = re.sub(r"<script\b.*?</script>", "", h, flags=re.S)
    h = re.sub(r"<style\b.*?</style>", "", h, flags=re.S)
    h = re.sub(r"<header class=\"site-header\">.*?</header>", "", h, flags=re.S)
    h = re.sub(r"<nav\b.*?</nav>", "", h, flags=re.S)
    h = re.sub(r"<head>.*?</head>", "", h, flags=re.S)
    h = re.sub(r"</(p|h1|h2|h3|div|li|figcaption|section)>", "\n\n", h)
    h = re.sub(r"<[^>]+>", "", h)
    h = re.sub(r"[ \t]+", " ", h)
    h = re.sub(r"\n\s*\n+", "\n\n", h)
    (O / (f.stem + ".txt")).write_text(h.strip())
