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
def pdf_to_images(pdf_path, out_dir, dpi=110, prefix="slide"):
    """Render each PDF page to a PNG. Returns the sorted list of paths.

    Tries PyMuPDF first (pip installable, no system dependency), then falls
    back to poppler's `pdftoppm`. Run this once, offline; commit the PNGs next
    to the notebook so students do not need either tool.

    PowerPoint / Keynote: export to PDF first, or
    `libreoffice --headless --convert-to pdf deck.pptx`.
    """
    os.makedirs(out_dir, exist_ok=True)

    try:
        import fitz                                   # PyMuPDF
        doc = fitz.open(pdf_path)
        zoom = dpi / 72.0
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            pix.save(os.path.join(out_dir, f"{prefix}_{i:03d}.png"))
        doc.close()
        return sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.png")))
    except ImportError:
        pass

    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), pdf_path,
                        os.path.join(out_dir, prefix)], check=True)
        return sorted(glob.glob(os.path.join(out_dir, f"{prefix}-*.png")))

    raise RuntimeError(
        "Need PyMuPDF or poppler to rasterise the PDF.\n"
        "  pip install pymupdf          (no system packages needed)\n"
        "  apt-get install poppler-utils  (provides pdftoppm)")


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
