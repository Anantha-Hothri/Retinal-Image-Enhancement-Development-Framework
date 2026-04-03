import { useState } from 'react';
import { 
  ReactCompareSlider, 
  ReactCompareSliderImage 
} from 'react-compare-slider';
import { ArrowLeftRight, ZoomIn, ZoomOut } from 'lucide-react';

interface ResultsComparisonProps {
  requestId: string;
  metrics: Record<string, any>;
}

export default function ResultsComparison({ requestId, metrics }: ResultsComparisonProps) {
  const [comparisonMode, setComparisonMode] = useState<'slider' | 'side-by-side'>('slider');
  const [zoom, setZoom] = useState(1);

  const inputUrl = `http://localhost:8000/api/result/${requestId}/input`;
  const finalUrl = `http://localhost:8000/api/result/${requestId}/final`;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-900">
          Before & After Comparison
        </h2>
        
        {/* View Mode Toggle */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setComparisonMode('slider')}
            className={`
              px-4 py-2 rounded-lg font-medium transition-colors
              ${comparisonMode === 'slider'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}
            `}
          >
            Slider View
          </button>
          <button
            onClick={() => setComparisonMode('side-by-side')}
            className={`
              px-4 py-2 rounded-lg font-medium transition-colors
              ${comparisonMode === 'side-by-side'
                ? 'bg-indigo-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}
            `}
          >
            Side by Side
          </button>
        </div>
      </div>

      {/* Comparison Display */}
      <div className="mb-6">
        {comparisonMode === 'slider' ? (
          <div className="relative rounded-lg overflow-hidden shadow-md">
            <ReactCompareSlider
              itemOne={
                <ReactCompareSliderImage
                  src={inputUrl}
                  alt="Original Zeiss Visuscout"
                />
              }
              itemTwo={
                <ReactCompareSliderImage
                  src={finalUrl}
                  alt="Enhanced Clarus-quality"
                />
              }
              position={50}
              style={{ height: '600px' }}
            />
            {/* Labels */}
            <div className="absolute top-4 left-4 bg-black bg-opacity-60 text-white px-3 py-1 rounded-lg text-sm font-medium">
              Original
            </div>
            <div className="absolute top-4 right-4 bg-black bg-opacity-60 text-white px-3 py-1 rounded-lg text-sm font-medium">
              Enhanced
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <h3 className="font-semibold text-gray-700">Original Zeiss Visuscout</h3>
              <div className="relative rounded-lg overflow-hidden shadow-md">
                <img
                  src={inputUrl}
                  alt="Original"
                  className="w-full"
                  style={{ transform: `scale(${zoom})` }}
                />
              </div>
            </div>
            <div className="space-y-2">
              <h3 className="font-semibold text-gray-700">Enhanced Clarus-quality</h3>
              <div className="relative rounded-lg overflow-hidden shadow-md">
                <img
                  src={finalUrl}
                  alt="Enhanced"
                  className="w-full"
                  style={{ transform: `scale(${zoom})` }}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 gap-4 p-4 bg-gray-50 rounded-lg">
        <div className="text-center">
          <p className="text-sm text-gray-600">Original Size</p>
          <p className="text-lg font-semibold text-gray-900">
            {metrics.size_original || 'N/A'}
          </p>
        </div>
        <div className="text-center">
          <p className="text-sm text-gray-600">Enhanced Size</p>
          <p className="text-lg font-semibold text-gray-900">
            {metrics.size_enhanced || 'N/A'}
          </p>
        </div>
        <div className="text-center">
          <p className="text-sm text-gray-600">Resolution Improvement</p>
          <p className="text-lg font-semibold text-indigo-600">
            {metrics.resolution_improvement || 'N/A'}
          </p>
        </div>
      </div>

      {/* Download Buttons */}
      <div className="flex justify-center space-x-4 mt-6">
        <a
          href={finalUrl}
          download="enhanced_result.png"
          className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors font-medium"
        >
          Download Enhanced Image
        </a>
        <a
          href={inputUrl}
          download="original_input.jpg"
          className="bg-gray-200 text-gray-700 px-6 py-3 rounded-lg hover:bg-gray-300 transition-colors font-medium"
        >
          Download Original
        </a>
      </div>
    </div>
  );
}

