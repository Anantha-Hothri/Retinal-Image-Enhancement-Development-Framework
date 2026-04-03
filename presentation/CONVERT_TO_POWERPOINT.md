# Converting Markdown Slides to PowerPoint

## Option 1: Using Pandoc (Recommended)

**Install Pandoc:**
```bash
# macOS
brew install pandoc

# Windows
# Download from: https://pandoc.org/installing.html

# Linux
sudo apt-get install pandoc
```

**Convert to PowerPoint:**
```bash
cd /Users/inuguria/Documents/FYP/augment/presentation

# Basic conversion
pandoc PRESENTATION_SLIDES.md -o presentation.pptx

# With custom theme (optional)
pandoc PRESENTATION_SLIDES.md \
  --reference-doc=template.pptx \
  -o presentation.pptx
```

---

## Option 2: Using Marp (Markdown Presentation)

**Install Marp CLI:**
```bash
npm install -g @marp-team/marp-cli
```

**Convert to PowerPoint:**
```bash
cd /Users/inuguria/Documents/FYP/augment/presentation

marp PRESENTATION_SLIDES.md --pptx -o presentation.pptx
```

**Convert to PDF:**
```bash
marp PRESENTATION_SLIDES.md --pdf -o presentation.pdf
```

---

## Option 3: Using reveal-md (HTML Slides)

**Install reveal-md:**
```bash
npm install -g reveal-md
```

**View as Interactive Slides:**
```bash
cd /Users/inuguria/Documents/FYP/augment/presentation

reveal-md PRESENTATION_SLIDES.md
# Opens in browser at http://localhost:1948
```

**Export to PDF:**
```bash
reveal-md PRESENTATION_SLIDES.md --print presentation.pdf
```

---

## Option 4: Manual Import to Google Slides

1. Go to https://slides.google.com
2. Create new presentation
3. File → Import slides
4. Upload the PDF version (created with Marp or Pandoc)
5. Select slides to import
6. Edit and format as needed

---

## Option 5: Use Online Converters

**Recommended Online Tools:**
- **Slides.com** - https://slides.com
- **Beautiful.ai** - https://www.beautiful.ai
- **Pitch** - https://pitch.com

**Steps:**
1. Copy content from PRESENTATION_SLIDES.md
2. Paste into online editor
3. Format and design
4. Download as PPTX or PDF

---

## Recommended Workflow

**For Quick Conversion:**
```bash
# Install Pandoc
brew install pandoc

# Convert to PowerPoint
cd /Users/inuguria/Documents/FYP/augment/presentation
pandoc PRESENTATION_SLIDES.md -o FYP_Presentation.pptx

# Convert to PDF (backup)
pandoc PRESENTATION_SLIDES.md -t beamer -o FYP_Presentation.pdf
```

**For Best Results:**
1. Use Pandoc to create initial PPTX
2. Open in PowerPoint/Keynote/Google Slides
3. Add diagrams from Mermaid renders (already displayed)
4. Insert actual images from processing results
5. Apply custom theme/styling
6. Add animations/transitions

---

## Adding Images to Presentation

**Images to Include:**

1. **System Architecture Diagram**
   - Already rendered as Mermaid diagram
   - Screenshot and insert

2. **Enhancement Pipeline**
   - Already rendered as Mermaid diagram
   - Screenshot and insert

3. **Training Loop Diagram**
   - Already rendered as Mermaid diagram
   - Screenshot and insert

4. **SFT-ESRGAN Architecture**
   - Already rendered as Mermaid diagram
   - Screenshot and insert

5. **Actual Processing Results**
   ```bash
   # Use outputs from test processing
   uploads/results/
   ├── 00_original.png
   ├── 01_dehazed.png
   ├── 02_clahe.png
   ├── 03_vessel_map.png
   ├── 04_enhanced.png
   └── 05_final.png
   ```

6. **Training Progress**
   - TensorBoard screenshots
   - Loss curves
   - SSIM/PSNR graphs

---

## Quick Commands Summary

```bash
# Navigate to presentation folder
cd /Users/inuguria/Documents/FYP/augment/presentation

# Option A: Pandoc (most compatible)
pandoc PRESENTATION_SLIDES.md -o FYP_Presentation.pptx

# Option B: Marp (better design)
npm install -g @marp-team/marp-cli
marp PRESENTATION_SLIDES.md --pptx -o FYP_Presentation.pptx

# Option C: PDF for printing
pandoc PRESENTATION_SLIDES.md -t beamer -o FYP_Presentation.pdf
```

---

## File Locations

**Markdown Source:**
- `presentation/PRESENTATION_SLIDES.md` (main content)

**Diagrams:**
- Already rendered in your IDE (screenshot them)
- System Architecture
- Enhancement Pipeline  
- Training Loop
- SFT-ESRGAN Architecture

**Output Will Be:**
- `presentation/FYP_Presentation.pptx` (PowerPoint)
- `presentation/FYP_Presentation.pdf` (PDF backup)

---

## Tips for Best Results

**Before Converting:**
1. ✅ Review all content in PRESENTATION_SLIDES.md
2. ✅ Make sure diagrams are rendered
3. ✅ Prepare actual images from test runs
4. ✅ Check code snippets are readable

**After Converting:**
1. Open PPTX in PowerPoint/Keynote
2. Apply consistent theme/colors
3. Add your name, date, supervisor info
4. Insert screenshots of diagrams
5. Add real result images
6. Set slide transitions
7. Rehearse timing (~1 min per slide)

**Testing:**
1. Present to a friend/colleague
2. Check all images display correctly
3. Verify code is readable
4. Test on presentation computer
5. Have PDF backup ready

---

## Troubleshooting

**Issue: Pandoc not found**
```bash
# Install Pandoc
brew install pandoc  # macOS
# or download from pandoc.org
```

**Issue: Code blocks not formatting well**
- Use syntax highlighting in PowerPoint
- Or convert code blocks to images
- Use monospace font (Courier, Consolas)

**Issue: Diagrams not showing**
- Take screenshots of rendered Mermaid diagrams
- Insert as images in PowerPoint
- Alternative: Export diagrams as SVG/PNG

**Issue: File too large**
- Compress images before inserting
- Use PNG instead of uncompressed formats
- Reduce diagram resolution if needed

---

## Ready-to-Use Command

**Single command to create presentation:**
```bash
cd /Users/inuguria/Documents/FYP/augment/presentation && \
pandoc PRESENTATION_SLIDES.md \
  -o FYP_Presentation_$(date +%Y%m%d).pptx && \
echo "✅ Presentation created: FYP_Presentation_$(date +%Y%m%d).pptx"
```

This creates a timestamped PowerPoint file ready for editing!


