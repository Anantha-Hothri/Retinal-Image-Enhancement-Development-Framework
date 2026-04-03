import { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import ProcessingSteps from '../components/ProcessingSteps';
import ResultsComparison from '../components/ResultsComparison';
import ComparisonViewer from '../components/ComparisonViewer';
import MetricsDashboard from '../components/MetricsDashboard';
import ClarusUpload from '../components/ClarusUpload';
import ImageControls, { ImageAdjustments } from '../components/ImageControls';

export default function Home() {
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [clarusImage, setClarusImage] = useState<File | null>(null);
  const [clarusMetrics, setClarusMetrics] = useState<any>(null);
  const [comparingClarus, setComparingClarus] = useState(false);
  const [imageAdjustments, setImageAdjustments] = useState<ImageAdjustments>({
    brightness: 100,
    contrast: 100,
    zoom: 100,
    rotation: 0,
  });

  const handleImageControlsChange = (adjustments: ImageAdjustments) => {
    setImageAdjustments(adjustments);
  };

  const handleImageUpload = async (file: File) => {
    setProcessing(true);
    setError(null);
    setResults(null);
    setClarusImage(null);
    setClarusMetrics(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Processing failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setProcessing(false);
    }
  };

  const handleClarusUpload = async (file: File) => {
    if (!results || !results.request_id) {
      setError('No enhanced image available for comparison');
      return;
    }

    setComparingClarus(true);
    setError(null);

    try {
      // Get the enhanced image from the backend
      const enhancedResponse = await fetch(
        `http://localhost:8000/api/result/${results.request_id}/final`
      );

      if (!enhancedResponse.ok) {
        throw new Error('Failed to fetch enhanced image');
      }

      const enhancedBlob = await enhancedResponse.blob();
      const enhancedFile = new File([enhancedBlob], 'enhanced.png', { type: 'image/png' });

      // Compare enhanced image with Clarus ground truth
      const formData = new FormData();
      formData.append('file1', enhancedFile);  // Enhanced result
      formData.append('file2', file);  // Clarus ground truth

      const compareResponse = await fetch('http://localhost:8000/api/compare', {
        method: 'POST',
        body: formData,
      });

      if (!compareResponse.ok) {
        throw new Error('Comparison failed');
      }

      const compareData = await compareResponse.json();

      if (compareData.success) {
        setClarusImage(file);
        setClarusMetrics(compareData.metrics);
        console.log('Clarus comparison metrics:', compareData.metrics);
      } else {
        throw new Error('Comparison returned unsuccessful status');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Comparison failed');
      console.error('Clarus comparison error:', err);
    } finally {
      setComparingClarus(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900">
            Retinal Image Enhancement
          </h1>
          <p className="mt-2 text-sm text-gray-600">
            Transform low-quality Zeiss Visuscout images to Zeiss Clarus quality using AI
          </p>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Upload Section */}
        {!results && (
          <div className="mb-8">
            <ImageUpload onUpload={handleImageUpload} disabled={processing} />
            {processing && (
              <div className="mt-4 text-center">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                <p className="mt-2 text-gray-600">Processing image...</p>
              </div>
            )}
            {error && (
              <div className="mt-4 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
                Error: {error}
              </div>
            )}
          </div>
        )}

        {/* Results Section */}
        {results && !processing && (
          <div className="space-y-8">
            {/* Processing Steps */}
            <ProcessingSteps
              steps={results.steps}
              requestId={results.request_id}
            />

            {/* Metrics Dashboard */}
            <MetricsDashboard metrics={results.metrics} />

            {/* Advanced Comparison Viewer */}
            <ComparisonViewer
              originalUrl={`http://localhost:8000/api/result/${results.request_id}/input`}
              enhancedUrl={`http://localhost:8000/api/result/${results.request_id}/final`}
              imageAdjustments={imageAdjustments}
            />

            {/* Image Controls */}
            <ImageControls onControlsChange={handleImageControlsChange} />

            {/* Optional Clarus Upload and Comparison */}
            {!clarusImage && (
              <ClarusUpload
                onUpload={handleClarusUpload}
                disabled={comparingClarus}
              />
            )}

            {comparingClarus && (
              <div className="text-center bg-white rounded-lg shadow-lg p-6">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600"></div>
                <p className="mt-2 text-gray-600">Comparing with Clarus ground truth...</p>
              </div>
            )}

            {/* Clarus Comparison Metrics */}
            {clarusImage && clarusMetrics && (
              <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-4">
                  Ground Truth Comparison
                </h2>
                <p className="text-sm text-gray-600 mb-6">
                  Comparison between enhanced result and Clarus ground truth image
                </p>
                <MetricsDashboard metrics={clarusMetrics} />
              </div>
            )}

            {/* Reset Button */}
            <div className="text-center">
              <button
                onClick={() => {
                  setResults(null);
                  setClarusImage(null);
                  setClarusMetrics(null);
                  setError(null);
                }}
                className="bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 transition-colors"
              >
                Process Another Image
              </button>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-12 bg-white border-t border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            Medical Image Enhancement System • FYP 2026
          </p>
        </div>
      </footer>
    </div>
  );
}

