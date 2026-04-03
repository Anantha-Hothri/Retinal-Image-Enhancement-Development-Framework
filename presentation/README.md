# Retinal Image Enhancement FYP Presentation

## 📋 Overview
This is a professional academic LaTeX Beamer presentation for the Final Year Project (FYP) on **Retinal Image Enhancement: Zeiss to Clarus Quality**.

## 🚀 Quick Start

### Compile on Overleaf (Recommended)
1. Go to [Overleaf.com](https://www.overleaf.com)
2. Create a new blank project
3. Upload `main.tex` to the project
4. Click **Recompile** to generate the PDF
5. The presentation is ready!

### Compile Locally
```bash
# Using pdflatex (run twice for proper ToC)
pdflatex main.tex
pdflatex main.tex

# Or using latexmk (automatic)
latexmk -pdf main.tex
```

## 📸 Adding Your Images

### Required Images
Replace the placeholder image paths with your actual retinal images:

1. **`original_zeiss.png`** - Original low-quality Zeiss input image
2. **`enhanced_output.png`** - Enhanced output from the 6-step pipeline (05_final.png)
3. **`clarus_gt.png`** - Clarus ground truth reference image
4. **`overlay_visualization.png`** - Overlay of enhanced output on Clarus ground truth

### Image Location
- Place images in the same directory as `main.tex`, or
- Create an `images/` folder and update paths: `\includegraphics{images/original_zeiss.png}`

### Recommended Images from Your Dataset
Use images from **Patient 10** (LE DC.JPG) as they showed excellent registration:
- **Original:** `patient_images/Patient 10/LE DC.JPG`
- **Enhanced:** `temp/patient10_LEDC/05_final.png`
- **Clarus GT:** `patient_images/Patient 10/...` (corresponding Clarus file)
- **Overlay:** `temp/patient10_LEDC/ground_truth_comparison/overlay_enhanced_on_clarus.png`

## ✏️ Customization

### Update Personal Information
Edit these sections in `main.tex`:

**Line 19-21 (Title slide):**
```latex
\author{Your Name}
\institute{Your University\\Department of Computer Science/Engineering}
```

**Line 437-439 (Contact slide):**
```latex
Your Name\\
your.email@university.edu\\
GitHub: github.com/yourusername/project-repo
```

### Adjust Metrics (if needed)
If you have different quality metrics, update the table on **Line 228-238**:
```latex
\textbf{PSNR (dB)} & 18.5 & \textcolor{green!50!black}{\textbf{28.7}} & ...
```

## 📊 Presentation Structure

1. **Title Slide** - Project title and your name
2. **Outline** - Automatic table of contents
3. **Introduction** - Clinical context and diabetic retinopathy
4. **Problem Statement** - Camera comparison and vessel classification
5. **Methodology** - 6-step enhancement pipeline and ground truth comparison
6. **Results** - Quality metrics, visual comparisons, overlay visualization
7. **Technical Implementation** - System architecture
8. **Challenges & Solutions** - Key problems overcome
9. **Conclusion** - Achievements, clinical significance, future work
10. **Questions** - Thank you and contact information

## 🎨 Theme Information
- **Theme:** Madrid (professional academic style)
- **Aspect Ratio:** 16:9 (widescreen)
- **Color Scheme:** Blue/Red/Green for different elements

## 📝 Presentation Tips

### During Presentation
1. **Introduction (Slide 3):** Emphasize the clinical need - diabetic retinopathy is a serious global health issue
2. **Camera Comparison (Slide 4):** Highlight the huge FOV difference (133° vs 40°)
3. **Vessel Classification (Slide 5):** Explain why tertiary vessels matter for diagnosis
4. **Pipeline (Slide 7):** Briefly walk through each step, mention GPU acceleration
5. **Results (Slide 9):** Focus on the significant improvements in metrics
6. **Overlay (Slide 11):** Show anatomical alignment demonstrates structure preservation
7. **Challenges (Slide 13):** Demonstrate problem-solving ability

### Time Management (15-20 min presentation)
- Introduction: 2-3 min
- Problem Statement: 3-4 min  
- Methodology: 4-5 min
- Results: 3-4 min
- Technical Implementation: 2-3 min
- Challenges: 2 min
- Conclusion: 2 min
- Questions: Remainder

## 🐛 Troubleshooting

### Images Not Showing
- Ensure image files are in the correct directory
- Check file extensions match exactly (case-sensitive on Linux)
- Verify image paths in `\includegraphics{...}`

### Compilation Errors
- Make sure all required packages are installed
- Try compiling twice (LaTeX needs two passes for ToC)
- Check for missing `\end{...}` statements

### Special Characters
- If you see encoding issues, add: `\usepackage[utf8]{inputenc}`

## 📦 Files Included
- `main.tex` - Main presentation file (ready to compile)
- `README.md` - This file (usage instructions)

## 🎯 Next Steps
1. Upload to Overleaf and compile
2. Add your actual retinal images
3. Update your name and contact information
4. Practice your presentation timing
5. Prepare for questions about methodology and results

Good luck with your presentation! 🚀

