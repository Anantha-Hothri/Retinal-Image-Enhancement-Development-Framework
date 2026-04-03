import { useState } from 'react';
import { Sun, Contrast, Maximize, RotateCcw } from 'lucide-react';

interface ImageControlsProps {
  onControlsChange?: (controls: ImageAdjustments) => void;
}

export interface ImageAdjustments {
  brightness: number;
  contrast: number;
  zoom: number;
  rotation: number;
}

export default function ImageControls({ onControlsChange }: ImageControlsProps) {
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [zoom, setZoom] = useState(100);
  const [rotation, setRotation] = useState(0);

  const handleChange = (
    type: keyof ImageAdjustments,
    value: number
  ) => {
    const updates: { [key: string]: (value: number) => void } = {
      brightness: setBrightness,
      contrast: setContrast,
      zoom: setZoom,
      rotation: setRotation,
    };
    
    updates[type](value);
    
    if (onControlsChange) {
      onControlsChange({
        brightness: type === 'brightness' ? value : brightness,
        contrast: type === 'contrast' ? value : contrast,
        zoom: type === 'zoom' ? value : zoom,
        rotation: type === 'rotation' ? value : rotation,
      });
    }
  };

  const handleReset = () => {
    setBrightness(100);
    setContrast(100);
    setZoom(100);
    setRotation(0);
    
    if (onControlsChange) {
      onControlsChange({
        brightness: 100,
        contrast: 100,
        zoom: 100,
        rotation: 0,
      });
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-gray-900">
          Medical Image Controls
        </h3>
        <button
          onClick={handleReset}
          className="flex items-center space-x-2 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
        >
          <RotateCcw className="w-4 h-4" />
          <span>Reset</span>
        </button>
      </div>

      <div className="space-y-6">
        {/* Brightness Control */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Sun className="w-5 h-5 text-yellow-600" />
              <label className="text-sm font-medium text-gray-700">
                Brightness
              </label>
            </div>
            <span className="text-sm font-semibold text-gray-900">
              {brightness}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="200"
            value={brightness}
            onChange={(e) => handleChange('brightness', Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Contrast Control */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Contrast className="w-5 h-5 text-blue-600" />
              <label className="text-sm font-medium text-gray-700">
                Contrast
              </label>
            </div>
            <span className="text-sm font-semibold text-gray-900">
              {contrast}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="200"
            value={contrast}
            onChange={(e) => handleChange('contrast', Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Zoom Control */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Maximize className="w-5 h-5 text-green-600" />
              <label className="text-sm font-medium text-gray-700">
                Zoom
              </label>
            </div>
            <span className="text-sm font-semibold text-gray-900">
              {zoom}%
            </span>
          </div>
          <input
            type="range"
            min="50"
            max="300"
            value={zoom}
            onChange={(e) => handleChange('zoom', Number(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* Info Box */}
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>Clinical Note:</strong> These controls allow medical professionals to adjust image parameters for better visibility of retinal structures. All adjustments are non-destructive and can be reset.
          </p>
        </div>
      </div>
    </div>
  );
}

