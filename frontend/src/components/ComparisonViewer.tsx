import { useState, useEffect, useRef } from 'react';
import {
  ReactCompareSlider,
  ReactCompareSliderImage
} from 'react-compare-slider';
import { Layers, SplitSquareHorizontal, Activity } from 'lucide-react';

interface ImageAdjustments {
  brightness: number;
  contrast: number;
  zoom: number;
  rotation: number;
}

interface ComparisonViewerProps {
  originalUrl: string;
  enhancedUrl: string;
  clarusUrl?: string; // Optional ground truth
  imageAdjustments?: ImageAdjustments;
}

export default function ComparisonViewer({
  originalUrl,
  enhancedUrl,
  clarusUrl,
  imageAdjustments
}: ComparisonViewerProps) {
  // Default adjustments if not provided
  const adjustments = imageAdjustments || {
    brightness: 100,
    contrast: 100,
    zoom: 100,
    rotation: 0
  };

  // Create CSS filter string from adjustments
  const getImageStyle = () => {
    return {
      filter: `brightness(${adjustments.brightness}%) contrast(${adjustments.contrast}%)`,
      transform: `scale(${adjustments.zoom / 100}) rotate(${adjustments.rotation}deg)`,
      transition: 'filter 0.2s ease, transform 0.2s ease'
    };
  };
  const [viewMode, setViewMode] = useState<'slider' | 'side-by-side' | 'overlay' | 'difference'>('slider');
  const [opacity, setOpacity] = useState(50);
  const [showDifference, setShowDifference] = useState(false);
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null);
  const [generatingHeatmap, setGeneratingHeatmap] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Generate heatmap when difference mode is activated
  useEffect(() => {
    if (viewMode === 'difference' && !heatmapUrl && !generatingHeatmap) {
      generateHeatmap();
    }
  }, [viewMode]);

  const generateHeatmap = async () => {
    setGeneratingHeatmap(true);
    try {
      // Load both images
      const [img1, img2] = await Promise.all([
        loadImage(originalUrl),
        loadImage(enhancedUrl)
      ]);

      // Create canvas for heatmap generation
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Use the larger image dimensions
      canvas.width = Math.max(img1.width, img2.width);
      canvas.height = Math.max(img1.height, img2.height);

      // Draw both images to get pixel data
      const tempCanvas1 = document.createElement('canvas');
      const tempCanvas2 = document.createElement('canvas');
      tempCanvas1.width = canvas.width;
      tempCanvas1.height = canvas.height;
      tempCanvas2.width = canvas.width;
      tempCanvas2.height = canvas.height;

      const ctx1 = tempCanvas1.getContext('2d');
      const ctx2 = tempCanvas2.getContext('2d');

      if (!ctx1 || !ctx2) return;

      // Draw images (resize if needed)
      ctx1.drawImage(img1, 0, 0, canvas.width, canvas.height);
      ctx2.drawImage(img2, 0, 0, canvas.width, canvas.height);

      // Get pixel data
      const imageData1 = ctx1.getImageData(0, 0, canvas.width, canvas.height);
      const imageData2 = ctx2.getImageData(0, 0, canvas.width, canvas.height);
      const diffData = ctx.createImageData(canvas.width, canvas.height);

      // Calculate pixel-wise difference and create heatmap
      for (let i = 0; i < imageData1.data.length; i += 4) {
        // Calculate grayscale difference
        const r1 = imageData1.data[i];
        const g1 = imageData1.data[i + 1];
        const b1 = imageData1.data[i + 2];

        const r2 = imageData2.data[i];
        const g2 = imageData2.data[i + 1];
        const b2 = imageData2.data[i + 2];

        // L1 difference in RGB space
        const diff = (Math.abs(r1 - r2) + Math.abs(g1 - g2) + Math.abs(b1 - b2)) / 3;

        // Map difference to heatmap color (blue = no difference, red = large difference)
        const normalized = Math.min(diff / 128, 1); // Normalize to [0, 1]

        // Apply jet colormap
        const [r, g, b] = jetColormap(normalized);

        diffData.data[i] = r;
        diffData.data[i + 1] = g;
        diffData.data[i + 2] = b;
        diffData.data[i + 3] = 255; // Alpha
      }

      ctx.putImageData(diffData, 0, 0);

      // Convert canvas to blob URL
      canvas.toBlob((blob) => {
        if (blob) {
          const url = URL.createObjectURL(blob);
          setHeatmapUrl(url);
        } else {
          console.error('Failed to create blob from canvas');
        }
      });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      console.error('Error generating heatmap:', errorMessage, error);
    } finally {
      setGeneratingHeatmap(false);
    }
  };

  const loadImage = (url: string): Promise<HTMLImageElement> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(new Error(`Failed to load image from ${url}`));
      img.src = url;
    });
  };

  // Jet colormap function (blue -> cyan -> green -> yellow -> red)
  const jetColormap = (value: number): [number, number, number] => {
    const v = Math.max(0, Math.min(1, value));

    let r, g, b;

    if (v < 0.25) {
      r = 0;
      g = v * 4 * 255;
      b = 255;
    } else if (v < 0.5) {
      r = 0;
      g = 255;
      b = (1 - (v - 0.25) * 4) * 255;
    } else if (v < 0.75) {
      r = (v - 0.5) * 4 * 255;
      g = 255;
      b = 0;
    } else {
      r = 255;
      g = (1 - (v - 0.75) * 4) * 255;
      b = 0;
    }

    return [Math.round(r), Math.round(g), Math.round(b)];
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">
          Comparison Viewer
        </h2>
        
        {/* View Mode Selector */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setViewMode('slider')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'slider'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
            title="Interactive Slider"
          >
            <SplitSquareHorizontal className="w-4 h-4" />
            <span>Slider</span>
          </button>
          
          <button
            onClick={() => setViewMode('side-by-side')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'side-by-side'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
            title="Side by Side"
          >
            <Layers className="w-4 h-4" />
            <span>Side by Side</span>
          </button>
          
          <button
            onClick={() => setViewMode('overlay')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'overlay'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
            title="Overlay with Opacity"
          >
            <Layers className="w-4 h-4" />
            <span>Overlay</span>
          </button>
          
          <button
            onClick={() => setViewMode('difference')}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'difference'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
            title="Difference Heatmap"
          >
            <Activity className="w-4 h-4" />
            <span>Difference</span>
          </button>
        </div>
      </div>

      {/* Viewer Content */}
      <div className="mb-6">
        {viewMode === 'slider' && (
          <div className="relative rounded-lg overflow-hidden shadow-md">
            <div style={getImageStyle()}>
              <ReactCompareSlider
                itemOne={<ReactCompareSliderImage src={originalUrl} alt="Original" />}
                itemTwo={<ReactCompareSliderImage src={enhancedUrl} alt="Enhanced" />}
                position={50}
                style={{ height: '600px' }}
              />
            </div>
            <div className="absolute top-4 left-4 bg-black bg-opacity-60 text-white px-3 py-1 rounded-lg text-sm font-medium">
              Original Zeiss
            </div>
            <div className="absolute top-4 right-4 bg-black bg-opacity-60 text-white px-3 py-1 rounded-lg text-sm font-medium">
              Enhanced
            </div>
          </div>
        )}

        {viewMode === 'side-by-side' && (
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <h3 className="font-semibold text-gray-700">Original Zeiss Visuscout</h3>
              <img
                src={originalUrl}
                alt="Original"
                className="w-full rounded-lg shadow-md"
                style={getImageStyle()}
              />
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-gray-700">Enhanced Result</h3>
              <img
                src={enhancedUrl}
                alt="Enhanced"
                className="w-full rounded-lg shadow-md"
                style={getImageStyle()}
              />
            </div>
          </div>
        )}

        {viewMode === 'overlay' && (
          <div className="space-y-4">
            <div className="relative rounded-lg overflow-hidden shadow-md" style={{ height: '600px' }}>
              <img
                src={originalUrl}
                alt="Original"
                className="absolute inset-0 w-full h-full object-contain"
                style={getImageStyle()}
              />
              <img
                src={enhancedUrl}
                alt="Enhanced"
                className="absolute inset-0 w-full h-full object-contain"
                style={{
                  opacity: opacity / 100,
                  ...getImageStyle()
                }}
              />
            </div>

            {/* Opacity Slider */}
            <div className="flex items-center space-x-4">
              <span className="text-sm font-medium text-gray-700 w-24">Opacity:</span>
              <input
                type="range"
                min="0"
                max="100"
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
              <span className="text-sm font-medium text-gray-700 w-12">{opacity}%</span>
            </div>
          </div>
        )}

        {viewMode === 'difference' && (
          <div className="space-y-4">
            {generatingHeatmap && (
              <div className="bg-gray-100 rounded-lg p-8 text-center">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
                <p className="text-gray-600">Generating difference heatmap...</p>
              </div>
            )}

            {!generatingHeatmap && heatmapUrl && (
              <div className="space-y-4">
                <div className="relative rounded-lg overflow-hidden shadow-md">
                  <img
                    src={heatmapUrl}
                    alt="Difference Heatmap"
                    className="w-full"
                  />
                  <div className="absolute top-4 left-4 bg-black bg-opacity-70 text-white px-4 py-2 rounded-lg">
                    <p className="text-sm font-semibold">Pixel-wise Difference Heatmap</p>
                  </div>
                </div>

                {/* Colormap Legend */}
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h4 className="text-sm font-semibold text-gray-900 mb-3">Color Scale</h4>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-600 whitespace-nowrap">Low Diff</span>
                    <div className="flex-1 h-6 rounded" style={{
                      background: 'linear-gradient(to right, #0000FF, #00FFFF, #00FF00, #FFFF00, #FF0000)'
                    }}></div>
                    <span className="text-xs text-gray-600 whitespace-nowrap">High Diff</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    Blue regions show similar pixels, red regions show large differences
                  </p>
                </div>
              </div>
            )}

            {!generatingHeatmap && !heatmapUrl && (
              <div className="bg-gray-100 rounded-lg p-8 text-center">
                <Activity className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600">
                  Failed to generate heatmap
                </p>
                <button
                  onClick={generateHeatmap}
                  className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
                >
                  Retry
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

