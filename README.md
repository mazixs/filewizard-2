# File Wizard

[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue?logo=paypal&logoColor=white)](https://www.paypal.me/unterrikermanu)
[![Docker Pulls](https://img.shields.io/docker/pulls/loredcast/filewizard.svg)](https://hub.docker.com/r/loredcast/filewizard)
[![Docker Image Version](https://img.shields.io/docker/v/loredcast/filewizard/0.4-latest.svg)](https://hub.docker.com/r/loredcast/filewizard)

A self-hosted, browser-based utility for file conversion, OCR and audio transcription. It wraps common CLI and Python converters (FFmpeg, LibreOffice, Pandoc, ImageMagick, etc.), plus `faster-whisper` and Tesseract OCR.

![Screenshot](screenshot.png)

## Features
- Convert between many file formats; extendable via `settings.yml` to add any CLI tool.
- OCR for PDFs and images (`tesseract` / `PyMuPDF`).
- Audio transcription using Whisper models.
- Simple, responsive dark UI with drag-and-drop and file picker.
- Background job processing with real-time status updates and persistent history.
- `/settings` page for configuring conversion tools and OAuth (runs without auth in local mode).
- CPU-only by default; a `-cuda` image is available for GPU use.

## Security
**Warning:** exposing this app publicly without authentication risks arbitrary code execution. Intended for local use or behind a properly configured OAuth/OIDC provider.

#### Tech stack
FastAPI, vanilla HTML/JS/CSS frontend, SQLAlchemy (SQLite), Huey task queue.

## Installation

### Quick start — Docker Compose (pre-built image)

```bash
curl -L https://github.com/LoredCast/filewizard/raw/main/docker-compose.yml -o docker-compose.yml
docker compose up -d
```

FileWizard will be available at `http://localhost:6969`.

### Build locally with Docker

The Dockerfile provides three multi-stage build targets:

| Target | Description | Size |
|---|---|---|
| `small-final` | Minimal deps, no TeX, no TTS models | ~1 GB |
| `full-final` | All converters, TTS, transcription | ~1.8 GB |
| `cuda-final` | Full + CUDA 12.6 + GPU support | ~4 GB |

```bash
# Small build (recommended for most users)
docker build --target small-final -t filewizard:local .
docker run -d -p 6969:8000 -e LOCAL_ONLY=true filewizard:local

# Full build
docker build --target full-final -t filewizard:full .

# CUDA build
docker build --target cuda-final -t filewizard:cuda .
```

Or use the helper script:

```bash
./scripts/run-docker.sh small-final 6969
```

For GPU support, uncomment the `deploy` block in `docker-compose.yml` and use the `cuda-final` target.

### Local development (no Docker)

```bash
# One-time setup
./scripts/setup.sh

# Production-like mode (gunicorn + huey consumer)
./run.sh

# Development mode (uvicorn reload + huey consumer)
./scripts/dev.sh
```

Requirements: Python 3.12+ and system packages (Tesseract, LibreOffice, Pandoc, FFmpeg, etc.).

## GPU / CUDA setup

For CUDA builds, ensure you have:
- NVIDIA Docker runtime installed (`nvidia-docker2` package)
- Compatible GPU with appropriate drivers
- Add the GPU configuration to `docker-compose.yml`:

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

For troubleshooting GPU issues:
1. Your GPU drivers support CUDA 12.6
2. The `nvidia-container-toolkit` is properly configured
3. Test NVIDIA setup with: `docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi`

## Configuration & docs
See the project Wiki for details and examples:  
https://github.com/LoredCast/filewizard/wiki

## Usage
1. Open `http://127.0.0.1:8000` (or the port you mapped).
2. Drag & drop or choose files.
3. Select action: Convert, OCR, or Transcribe.
4. Track job progress in the History table (updates automatically).

# Tools Table

| Tool | Common inputs (extensions / format names) | Common outputs (extensions / format names) | Notes |
|---|---|---|---|
| **LibreOffice (soffice)** | `.odt`, `.fodt`, `.ott`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotx`, `.rtf`, `.txt`, `.html/.htm/.xhtml`, `.xml`, `.sxw`, `.wps`, `.wpd`, `.abw`, `.pdb`, `.epub`, `.fb2`, `.lit`, `.lrf`, `.pages`, `.csv`, `.tsv`, `.xls`, `.xlsx`, `.xlsm`, `.ods`, `.sxc`, `.123`, `.dbf`, `.fb2` | `.pdf`, `.pdfa`, `.odt`, `.fodt`, `.doc`, `.docx`, `.rtf`, `.txt`, `.html/.htm`, `.xhtml`, `.epub`, `.svg`, `.png`, `.jpg/.jpeg`, `.pptx`, `.ppt`, `.odp`, `.xls`, `.xlsx`, `.ods`, `.csv`, `.dbf`, `.pdb`, `.fb2` | Good for office/document conversions; fidelity varies with complex layouts. |
| **Pandoc** | Markdown flavors (`.md`, `.markdown`), `.html/.htm`, LaTeX (`.tex`), `.rst`, `.docx`, `.odt`, `.epub`, `.ipynb`, `.opml`, `.adoc`/asciidoc, `.tex`, `.bib`/citation inputs | `.html/.html5`, `.xhtml`, `.latex/.tex`, `.pdf` (via LaTeX engine), `.docx`, `.odt`, `.epub`, `.md` (flavors), `.gfm`, `.rst`, `.pptx`, `.man`, `.mediawiki`, `.docbook` | Highly configurable via templates/filters; requires LaTeX for PDF output. |
| **Ghostscript (gs)** | `.ps`, `.eps`, `.pdf`, PostScript streams | `.pdf` (various compat levels incl PDF/A), `.ps`, `.eps`, raster images (`.png`, `.jpg`, `.tiff`, `.bmp`, `.pnm`) | Useful for PDF manipulations, rasterization, and producing PDF/A. |
| **Calibre (ebook-convert)** | `.epub`, `.mobi`, `.azw3`, `.azw`, `.fb2`, `.html`, `.docx`, `.doc`, `.rtf`, `.txt`, `.pdb`, `.lit`, `.tcr`, `.cbz`, `.cbr`, `.odt`, `.pdf` (input with caveats) | `.epub`, `.mobi` (legacy), `.azw3`, `.pdf`, `.docx`, `.rtf`, `.txt`, `.fb2`, `.htmlz`, `.pdb`, `.lrf`, `.lit`, `.tcr`, `.cbz`, `.cbr` | Excellent for ebook format conversions and metadata handling; PDF input/output fidelity varies. |
| **FFmpeg** | Containers & codecs: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.flv`, `.wmv`, `.mpg/.mpeg`, `.ts`, `.m2ts`, `.3gp`, audio: `.mp3`, `.wav`, `.aac/.m4a`, `.flac`, `.ogg`, `.opus`, image sequences (`.png`, `.jpg`, `.tiff`), HLS (`.m3u8`) | Wide set: `.mp4`, `.mkv`, `.mov`, `.webm`, `.avi`, `.flv`, `.mp3`, `.aac/.m4a`, `.wav`, `.flac`, `.ogg`, `.opus`, `.gif` (animated), `.ts`, elementary streams, many codec/container combos | Extremely versatile — audio/video transcoding, extraction, container changes, filters. Supported formats depend on build flags and linked libraries. |
| **libvips (vips)** | `.jpg/.jpeg`, `.png`, `.tif/.tiff`, `.webp`, `.avif`, `.heif/.heic`, `.jp2`, `.gif` (frames), `.pnm`, `.fits`, `.exr`, PDF (via poppler delegate) | `.jpg/.jpeg`, `.png`, `.tif/.tiff`, `.webp`, `.avif`, `.heif`, `.jp2`, `.pnm`, `.v` (VIPS native), `.fits`, `.exr` | Fast, memory-efficient image processing; great for large images and tiling. |
| **GraphicsMagick (gm)** | `.jpg/.jpeg`, `.png`, `.gif`, `.tif/.tiff`, `.bmp`, `.ico`, `.eps`, `.pdf` (via Ghostscript/poppler), `.dpx`, `.pnm`, `.svg` (if delegate), `.webp` (if built), `.exr` | `.jpg/.jpeg`, `.png`, `.webp` (if enabled), `.tif/.tiff`, `.gif`, `.bmp`, `.pdf` (from images), `.eps`, `.ico`, `.xpm`, `.dpx` | Similar to ImageMagick but with different performance/behavior; supported formats depend on build/delegates. |
| **ImageMagick (convert / magick)** | Same as GraphicsMagick (large set; many delegates) | Same as GraphicsMagick | Often used interchangeably; watch for security considerations when processing untrusted images. |
| **Inkscape** | `.svg/.svgz`, `.pdf`, `.eps`, `.ps`, `.ai` (legacy imports), `.dxf`, raster images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.tiff`, `.bmp`) | `.svg`, `.pdf`, `.ps`, `.eps`, `.png`, `.emf`, `.wmf`, `.xaml`, `.dxf`, `.eps` | Vector editing and export; CLI useful for batch SVG → PNG/PDF conversions. |
| **libjxl (cjxl / djxl)** | Raster inputs: `.png`, `.jpg/.jpeg`, `.ppm/.pbm/.pgm`, `.gif`, etc. | `.jxl` (JPEG XL) | Encoder/decoder for JPEG XL; availability depends on build. |
| **resvg** | `.svg/.svgz` | `.png` (raster) | Fast, accurate SVG renderer — good for SVG→PNG conversion. |
| **Potrace** | Bitmaps: `.pbm`, `.pgm`, `.ppm` (PNM family), `.bmp` (via conversion) | Vector: `.svg`, `.pdf`, `.eps`, `.ps`, `.dxf`, `.geojson` | Traces bitmaps to vector paths; often used with pre-conversion steps. |
| **MarkItDown / markitdown** | `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.html`, `.eml`, `.msg`, `.md`, `.txt`, images, `.epub` | `.md` (Markdown) | Utility to extract/produce Markdown from various formats; implementation details vary. |
| **pngquant** | `.png` (truecolor/rgba) | `.png` (quantized palette PNG) | Lossy PNG quantization for smaller PNGs. |
| **MozJPEG (cjpeg, jpegtran)** | `.ppm/.pbm/.pgm` (PNM), `.bmp`, existing `.jpg` | `.jpg/.jpeg` (MozJPEG-optimized) | Produces smaller JPEGs with improved compression; good for recompression. |
| **SoX (Sound eXchange)** | `.wav`, `.aiff`, `.mp3` (if libmp3lame), `.flac`, `.ogg/.oga`, `.raw`, `.au`, `.voc`, `.w64`, `.gsm`, `.amr`, `.m4a` (if libs present) | `.wav`, `.aiff`, `.flac`, `.mp3`, `.ogg`, `.raw`, `.w64`, `.opus`, `.amr`, `.m4a` | Audio processing, normalization, effects; exact formats depend on linked libraries. |
| **Tesseract OCR** | Image formats (`.png`, `.jpg`, `.jpeg`, `.tiff`), PDFs (image PDFs) | Plain text (`.txt`), searchable PDF (PDF with text layer), HOCR, ALTO XML | OCR engine; language/training data required for best accuracy. |
| **faster-whisper / OpenAI Whisper (local)** | Audio: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac` | Plain text transcripts (`.txt`), `.srt`, `.vtt`, other subtitle formats | Local Whisper implementations for speech-to-text. Models and speed depend on CPU/GPU and model variant. |



Consider spending 30 seconds on the [» usage survey «](https://app.youform.com/forms/sdlb6mto) to help improve the app and suggest changes!


[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/loredcast)
---
