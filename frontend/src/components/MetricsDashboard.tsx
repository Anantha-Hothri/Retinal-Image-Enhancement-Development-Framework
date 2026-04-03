import { BarChart, TrendingUp, Image as ImageIcon, Maximize } from 'lucide-react';

interface MetricsDashboardProps {
  metrics: {
    resolution_improvement?: string;
    size_original?: string;
    size_enhanced?: string;
    psnr?: number;
    ssim?: number;
    vessel_recovery?: number;
  };
}

export default function MetricsDashboard({ metrics }: MetricsDashboardProps) {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6 flex items-center">
        <BarChart className="w-6 h-6 mr-2" />
        Quality Metrics
      </h2>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Resolution Improvement */}
        <div className="bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-lg p-6 border border-indigo-200">
          <div className="flex items-center justify-between mb-2">
            <Maximize className="w-5 h-5 text-indigo-600" />
            <span className="text-xs font-medium text-indigo-600 uppercase tracking-wide">
              Resolution
            </span>
          </div>
          <p className="text-3xl font-bold text-indigo-900 mb-1">
            {metrics.resolution_improvement || 'N/A'}
          </p>
          <p className="text-sm text-indigo-700">
            Improvement Factor
          </p>
        </div>

        {/* Original Size */}
        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6 border border-blue-200">
          <div className="flex items-center justify-between mb-2">
            <ImageIcon className="w-5 h-5 text-blue-600" />
            <span className="text-xs font-medium text-blue-600 uppercase tracking-wide">
              Original
            </span>
          </div>
          <p className="text-3xl font-bold text-blue-900 mb-1">
            {metrics.size_original || 'N/A'}
          </p>
          <p className="text-sm text-blue-700">
            Input Dimensions
          </p>
        </div>

        {/* Enhanced Size */}
        <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6 border border-green-200">
          <div className="flex items-center justify-between mb-2">
            <TrendingUp className="w-5 h-5 text-green-600" />
            <span className="text-xs font-medium text-green-600 uppercase tracking-wide">
              Enhanced
            </span>
          </div>
          <p className="text-3xl font-bold text-green-900 mb-1">
            {metrics.size_enhanced || 'N/A'}
          </p>
          <p className="text-sm text-green-700">
            Output Dimensions
          </p>
        </div>

        {/* PSNR (if available) */}
        {metrics.psnr !== undefined && (
          <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg p-6 border border-purple-200">
            <div className="flex items-center justify-between mb-2">
              <BarChart className="w-5 h-5 text-purple-600" />
              <span className="text-xs font-medium text-purple-600 uppercase tracking-wide">
                PSNR
              </span>
            </div>
            <p className="text-3xl font-bold text-purple-900 mb-1">
              {metrics.psnr.toFixed(2)} dB
            </p>
            <p className="text-sm text-purple-700">
              Peak Signal-to-Noise Ratio
            </p>
          </div>
        )}

        {/* SSIM (if available) */}
        {metrics.ssim !== undefined && (
          <div className="bg-gradient-to-br from-pink-50 to-pink-100 rounded-lg p-6 border border-pink-200">
            <div className="flex items-center justify-between mb-2">
              <BarChart className="w-5 h-5 text-pink-600" />
              <span className="text-xs font-medium text-pink-600 uppercase tracking-wide">
                SSIM
              </span>
            </div>
            <p className="text-3xl font-bold text-pink-900 mb-1">
              {metrics.ssim.toFixed(4)}
            </p>
            <p className="text-sm text-pink-700">
              Structural Similarity Index
            </p>
          </div>
        )}

        {/* Vessel Recovery (if available) */}
        {metrics.vessel_recovery !== undefined && (
          <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg p-6 border border-orange-200">
            <div className="flex items-center justify-between mb-2">
              <TrendingUp className="w-5 h-5 text-orange-600" />
              <span className="text-xs font-medium text-orange-600 uppercase tracking-wide">
                Vessel Recovery
              </span>
            </div>
            <p className="text-3xl font-bold text-orange-900 mb-1">
              {(metrics.vessel_recovery * 100).toFixed(1)}%
            </p>
            <p className="text-sm text-orange-700">
              Retinal Structure Preservation
            </p>
          </div>
        )}
      </div>

      {/* Information Note */}
      <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <p className="text-sm text-blue-800">
          <strong>Note:</strong> These metrics quantify the quality improvement achieved by the enhancement pipeline.
          Higher PSNR and SSIM values indicate better quality. Vessel Recovery measures how well retinal structures are preserved.
        </p>
      </div>
    </div>
  );
}

