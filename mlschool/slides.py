"""
mlschool.slides — show a slide deck inside a notebook, one slide at a time.

Motivation: you have a PDF/PowerPoint/Keynote deck for the lecture and you want
a few slides to appear *in the middle of a notebook*, advanced by clicking,
rather than switching windows.

There are three ways to do that, with different trade-offs:

1. `SlideDeck` (this module's default). Render the deck to one image per slide,
   then display them with an ipywidgets Prev/Next control. Works in JupyterLab,
   classic Notebook, VS Code and Google Colab; needs a live kernel. This is the
   most portable option and the one to use unless you have a reason not to.

2. `embed_pdf` / `embed_url`. Drop an <iframe> pointing at a PDF or at hosted
   slides (Google Slides, Speaker Deck, a reveal.js export). You get the native
   slide navigation and no image conversion. Requires the deck to be reachable
   by URL -- a *local* PDF will not render inside Colab, whose outputs are
   sandboxed on a different origin.

3. RISE / jupyterlab-rise. Turns the notebook ITSELF into a reveal.js slideshow.
   That is a different thing from what this module does: it presents your
   notebook as slides, rather than putting a slide inside your notebook. It also
   does not work in Colab.

Typical use::

    import mlschool as ms
    # once, offline:  ms.slides.pdf_to_images("lecture2.pdf", "slides/")
    deck = ms.slides.SlideDeck("slides/")        # width defaults to "100%"
    deck.show(start=12)          # drop this cell wherever you want the slides
"""

import glob
import os
import shutil
import subprocess

#: Global escape hatch. Set ``ms.slides.USE_STATIC = True`` to make every
#: ``SlideDeck.show()`` render plain images instead of an interactive widget.
#: Useful when exporting a notebook to HTML/PDF, and when a frontend cannot
#: display widgets at all.
USE_STATIC = False


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------
def _ink_bbox(img, tol=8):
    """Bounding box of everything that is not (near-)white."""
    from PIL import ImageChops, Image
    bg = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img.convert("RGB"), bg).convert("L")
    return diff.point(lambda v: 255 if v > tol else 0).getbbox()


def _shared_crop_box(images, margin=8, tol=8):
    """One crop box covering the content of EVERY page.

    Cropping each page to its own content would make the slides different
    sizes, so they would jump around as you click through. Taking the union
    keeps them identical and preserves the relative position of everything.
    """
    boxes = [b for b in (_ink_bbox(im, tol) for im in images) if b]
    if not boxes:
        return None
    l = max(min(b[0] for b in boxes) - margin, 0)
    t = max(min(b[1] for b in boxes) - margin, 0)
    r = min(max(b[2] for b in boxes) + margin, images[0].width)
    b_ = min(max(b[3] for b in boxes) + margin, images[0].height)
    return (l, t, r, b_)


def pdf_to_images(pdf_path, out_dir, dpi=110, prefix="slide", crop="auto",
                  margin=8, quiet=False):
    """Render each PDF page to a PNG. Returns the sorted list of paths.

    Run this once, offline; commit the PNGs so students need no PDF tooling.

    **crop** — decks are often "printed" to Letter/A4 rather than exported at
    slide size, which leaves the slide sitting in a band of white. The default
    ``"auto"`` measures where the ink actually is, across all pages, and trims
    every page to that single shared box. Pass ``crop=None`` to keep the pages
    exactly as they are, or a 4-tuple of *fractions* ``(left, top, right,
    bottom)`` to crop by hand.

    Backend: PyMuPDF if importable (``pip install pymupdf``), else poppler's
    ``pdftoppm``.

    PowerPoint / Keynote: export to PDF first, or
    ``libreoffice --headless --convert-to pdf deck.pptx``.
    """
    import io
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    pages = []

    try:
        import fitz                                       # PyMuPDF
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pages.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        doc.close()
    except ImportError:
        if not shutil.which("pdftoppm"):
            raise RuntimeError(
                "Need PyMuPDF or poppler to rasterise the PDF.\n"
                "  pip install pymupdf            (no system packages needed)\n"
                "  apt-get install poppler-utils  (provides pdftoppm)")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-png", "-r", str(dpi), pdf_path,
                            os.path.join(tmp, "p")], check=True)
            for f in sorted(glob.glob(os.path.join(tmp, "p*.png"))):
                pages.append(Image.open(f).convert("RGB"))

    if not pages:
        raise RuntimeError(f"no pages rendered from {pdf_path!r}")

    box = None
    if crop == "auto":
        box = _shared_crop_box(pages, margin=margin)
    elif crop:                                            # fractions
        w, h = pages[0].size
        l, t, r, b = crop
        box = (int(l * w), int(t * h), int(r * w), int(b * h))

    if box:
        before = pages[0].size
        pages = [im.crop(box) for im in pages]
        after = pages[0].size
        if not quiet and after != before:
            saved = 100 * (1 - (after[0] * after[1]) / (before[0] * before[1]))
            print(f"cropped {before[0]}x{before[1]} -> {after[0]}x{after[1]} "
                  f"(removed {saved:.0f}% whitespace, same box on every page)")

    paths = []
    for i, im in enumerate(pages):
        path = os.path.join(out_dir, f"{prefix}_{i:03d}.png")
        im.save(path)
        paths.append(path)
    if not quiet:
        print(f"wrote {len(paths)} slides to {out_dir} "
              f"({pages[0].width}x{pages[0].height}, "
              f"aspect {pages[0].width / pages[0].height:.2f})")
    return paths


def _css_width(width):
    """Accept 760 (pixels) or any CSS length string: "100%", "80vw", "40em"."""
    if isinstance(width, (int, float)):
        return f"{int(width)}px"
    return str(width)


# --------------------------------------------------------------------------
# the viewer
# --------------------------------------------------------------------------
class SlideDeck:
    """A clickable slide viewer for a directory (or list) of images.

    `width` may be a number of pixels (``760``) or any CSS length
    (``"100%"``, ``"80vw"``). ``"100%"`` makes the deck fill the notebook's
    output area and follow the window as it is resized, which is usually what
    you want when projecting. Height is left unset so the aspect ratio is
    preserved.
    """

    PATTERNS = ("*.png", "*.jpg", "*.jpeg", "*.svg")

    def __init__(self, images, width="100%"):
        if isinstance(images, str):
            found = self._scan(images)
            if not found:
                # Not found relative to the working directory. Try again relative
                # to the repository root: on Colab the notebook runs in /content
                # while the repo (and its slides/) sits in a subdirectory.
                from . import ROOT
                found = self._scan(os.path.join(str(ROOT), images))
            if not found:
                raise ValueError(
                    f"no slide images in {images!r} (looked there and in "
                    f"{os.path.join(str(__import__('mlschool').ROOT), images)!r})")
            images = found
        self.images = list(images)
        self.width = width
        if not self.images:
            raise ValueError("no slide images found")

    @classmethod
    def _scan(cls, directory):
        return sorted(f for pat in cls.PATTERNS
                      for f in glob.glob(os.path.join(directory, pat)))

    def __len__(self):
        return len(self.images)

    def _payload(self, i):
        path = self.images[i]
        with open(path, "rb") as fh:
            data = fh.read()
        return data, path.lower().endswith(".svg")

    def show(self, start=0, width=None, static=None):
        """Interactive viewer: Prev / Next buttons plus a slider.

        Requires a LIVE KERNEL. A widget is a live object owned by the kernel;
        the notebook file stores only a reference to it. So if you re-open a
        saved notebook, or restart the kernel, the browser reports

            Error displaying widget: model not found

        That is expected, and it is not a broken installation: just re-run the
        cell. If you need output that survives saving, use ``static=True`` (or
        set ``mlschool.slides.USE_STATIC = True`` once, for every deck).
        """
        if static or (static is None and USE_STATIC):
            return self.show_static([start], width=width)
        try:
            import ipywidgets as W
        except ImportError:
            print("ipywidgets not installed; falling back to static output.")
            return self.show_static([start], width=width)

        from IPython.display import display

        css = _css_width(width or self.width)
        # Set the CSS width via `layout`, NOT the `width` trait: the trait becomes
        # an HTML width attribute, which only accepts integers, so "100%" there is
        # silently ignored. height stays unset so the aspect ratio is preserved.
        img = W.Image(layout=W.Layout(width=css, height="auto", max_width="100%"))
        caption = W.HTML()
        slider = W.IntSlider(value=start, min=0, max=len(self) - 1,
                             description="slide", continuous_update=True,
                             layout=W.Layout(width=css, max_width="100%"))
        prev_b = W.Button(description="◀ Prev", layout=W.Layout(width="90px"))
        next_b = W.Button(description="Next ▶", layout=W.Layout(width="90px"))

        def render(i):
            data, is_svg = self._payload(i)
            img.format = "svg+xml" if is_svg else "png"
            img.value = data
            caption.value = (f"<div style='text-align:center;color:#666;"
                             f"font-family:sans-serif'>slide {i + 1} / "
                             f"{len(self)}</div>")

        slider.observe(lambda ch: render(ch["new"]) if ch["name"] == "value" else None)
        prev_b.on_click(lambda _b: setattr(slider, "value", max(0, slider.value - 1)))
        next_b.on_click(lambda _b: setattr(
            slider, "value", min(len(self) - 1, slider.value + 1)))

        render(start)
        display(W.VBox([img, caption,
                        W.HBox([prev_b, next_b, slider])],
                       layout=W.Layout(align_items="center", width=css,
                                       max_width="100%")))
        return None

    def show_static(self, indices=None, width=None):
        """Display selected slides as plain images (no interactivity).

        Use this when the notebook will be read rather than run -- the images
        survive into HTML/PDF exports and GitHub previews, which widgets do not.
        """
        from IPython.display import HTML, display
        css = _css_width(width or self.width)
        indices = range(len(self)) if indices is None else indices
        for i in indices:
            data, is_svg = self._payload(i)
            if is_svg:
                display(HTML(data.decode()))
            else:
                import base64
                b64 = base64.b64encode(data).decode()
                # style, not the width attribute, so percentages work here too
                display(HTML(f"<img src='data:image/png;base64,{b64}' "
                             f"style='width:{css};max-width:100%;height:auto'>"))


# --------------------------------------------------------------------------
# iframe embedding
# --------------------------------------------------------------------------
def embed_url(url, width="100%", height=560):
    """Embed hosted slides (Google Slides, reveal.js export, a PDF on the web).

    For Google Slides use *File -> Share -> Publish to the web -> Embed* and
    pass the `src` of the iframe it gives you. You get Google's own slide
    navigation, and it works in Colab because the content is same-protocol
    https rather than a local file.
    """
    from IPython.display import IFrame
    return IFrame(url, width=width, height=height)


def embed_pdf(path, width=900, height=560, page=1):
    """Embed a *local* PDF via an iframe, with the browser's own PDF viewer.

    Works in JupyterLab / classic Notebook / VS Code, where the file is served
    by the same origin as the notebook. **Does not work in Colab**, whose cell
    outputs run in a sandboxed iframe that cannot reach local files -- there,
    use `SlideDeck` or `embed_url`.
    """
    from IPython.display import IFrame
    return IFrame(f"{path}#page={page}", width=width, height=height)
