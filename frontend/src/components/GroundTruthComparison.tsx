import { useState, useEffect } from 'react';

interface GroundTruthComparisonProps {
  requestId: string;
}

interface ComparisonData {
  success: boolean;
  has_ground_truth: boolean;
  patient_id?: string;
  metrics?: {
    psnr: number;
    ssim: number;
    vessel_recovery: number;
    sharpness_zeiss: number;
    sharpness_enhanced: number;
    sharpness_clarus: number;
    contrast_zeiss: number;
    contrast_enhanced: number;
    contrast_clarus: number;
  };
  images?: {
    zeiss_original: string;
    enhanced_result: string;
    clarus_ground_truth: string;
    overlay: string;
  };
  message: string;
}

export default function GroundTruthComparison({ requestId }: GroundTruthComparisonProps) {
  const [loading, setLoading] = useState(false);
  const [comparisonData, setComparisonData] = useState<ComparisonData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (requestId) {
      loadGroundTruthComparison();
    }
  }, [requestId]);

  const loadGroundTruthComparison = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:8000/api/ground-truth-comparison/${requestId}`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to load ground truth comparison');
      }

      const data = await response.json();
      setComparisonData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Ground truth comparison error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="text-center bg-white rounded-lg shadow-lg p-8">
        <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600"></div>
        <p className="mt-4 text-gray-600">Loading ground truth comparison...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-red-800 mb-2">Error</h3>
        <p className="text-red-600">{error}</p>
      </div>
    );
  }

  if (!comparisonData || !comparisonData.has_ground_truth) {
    return (
      <div className="bg-yellow-50 rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-yellow-800 mb-2">No Ground Truth Available</h3>
        <p className="text-yellow-700">
          {comparisonData?.message || 'No matching Clarus ground truth image found in the dataset for this Zeiss image.'}
        </p>
      </div>
    );
  }

  const { metrics, images, patient_id } = comparisonData;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Ground Truth Comparison
        </h2>
        <p className="text-sm text-gray-600">
          Comparing enhanced result with Clarus ground truth image (Patient ID: {patient_id})
        </p>
      </div>

      {/* 3-Way Image Comparison */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Image Comparison</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Zeiss Original */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Original Zeiss</h4>
            <img
              src={`http://localhost:8000/api/result/${requestId}/${images?.zeiss_original}`}
              alt="Zeiss Original"
              className="w-full h-auto rounded border border-gray-300"
            />
          </div>

          {/* Enhanced Result */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Enhanced Result</h4>
            <img
              src={`http://localhost:8000/api/result/${requestId}/${images?.enhanced_result}?brightness=2.0`}
              alt="Enhanced Result"
              className="w-full h-auto rounded border border-gray-300"
            />
          </div>

          {/* Clarus Ground Truth */}
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-2">Clarus Ground Truth</h4>
            <img
              src={`http://localhost:8000/api/ground-truth-image/${requestId}/clarus_ground_truth`}
              alt="Clarus Ground Truth"
              className="w-full h-auto rounded border border-gray-300"
            />
          </div>
        </div>
      </div>

      {/* Overlay Visualization */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-semibold text-gray-900 mb-4">Overlay Visualization</h3>
        <p className="text-sm text-gray-600 mb-4">Zeiss image overlaid on Clarus ground truth (50% blend)</p>
        <div className="max-w-2xl mx-auto">
          <img
            src={`http://localhost:8000/api/ground-truth-image/${requestId}/overlay`}
            alt="Overlay Visualization"
            className="w-full h-auto rounded border border-gray-300"
          />
        </div>
      </div>

      {/* Metrics Dashboard */}
      {metrics && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Quality Metrics</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Comparison Metrics */}
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="text-sm font-semibold text-blue-900 mb-3">Enhanced vs. Ground Truth</h4>
              <div className="space-y-2">
                <div>
                  <span className="text-xs text-blue-700">PSNR</span>
                  <p className="text-lg font-bold text-blue-900">{metrics.psnr.toFixed(2)} dB</p>
                </div>
                <div>
                  <span className="text-xs text-blue-700">SSIM</span>
                  <p className="text-lg font-bold text-blue-900">{metrics.ssim.toFixed(4)}</p>
                </div>
                <div>
                  <span className="text-xs text-blue-700">Vessel Recovery</span>
                  <p className="text-lg font-bold text-blue-900">{(metrics.vessel_recovery * 100).toFixed(1)}%</p>
                </div>
              </div>
            </div>

            {/* Sharpness Comparison */}
            <div className="bg-green-50 p-4 rounded-lg">
              <h4 className="text-sm font-semibold text-green-900 mb-3">Sharpness</h4>
              <div className="space-y-2">
                <div>
                  <span className="text-xs text-green-700">Zeiss</span>
                  <p className="text-lg font-bold text-green-900">{metrics.sharpness_zeiss.toFixed(2)}</p>
                </div>
                <div>
                  <span className="text-xs text-green-700">Enhanced</span>
                  <p className="text-lg font-bold text-green-900">{metrics.sharpness_enhanced.toFixed(2)}</p>
                </div>
                <div>
                  <span className="text-xs text-green-700">Clarus</span>
                  <p className="text-lg font-bold text-green-900">{metrics.sharpness_clarus.toFixed(2)}</p>
                </div>
              </div>
            </div>

            {/* Contrast Comparison */}
            <div className="bg-purple-50 p-4 rounded-lg">
              <h4 className="text-sm font-semibold text-purple-900 mb-3">Contrast</h4>
              <div className="space-y-2">
                <div>
                  <span className="text-xs text-purple-700">Zeiss</span>
                  <p className="text-lg font-bold text-purple-900">{metrics.contrast_zeiss.toFixed(2)}</p>
                </div>
                <div>
                  <span className="text-xs text-purple-700">Enhanced</span>
                  <p className="text-lg font-bold text-purple-900">{metrics.contrast_enhanced.toFixed(2)}</p>
                </div>
                <div>
                  <span className="text-xs text-purple-700">Clarus</span>
                  <p className="text-lg font-bold text-purple-900">{metrics.contrast_clarus.toFixed(2)}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

