import { useState } from 'react';
import { ChevronRight } from 'lucide-react';

interface Step {
  name: string;
  title: string;
  description: string;
  path: string;
}

interface ProcessingStepsProps {
  steps: Step[];
  requestId: string;
}

export default function ProcessingSteps({ steps, requestId }: ProcessingStepsProps) {
  const [selectedStep, setSelectedStep] = useState(0);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">
        Processing Pipeline
      </h2>

      {/* Step Navigation */}
      <div className="flex items-center justify-between mb-8 overflow-x-auto">
        {steps.map((step, index) => (
          <div key={step.name} className="flex items-center">
            <button
              onClick={() => setSelectedStep(index)}
              className={`
                flex items-center space-x-2 px-4 py-2 rounded-lg transition-all
                ${selectedStep === index 
                  ? 'bg-indigo-600 text-white shadow-md' 
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}
              `}
            >
              <span className="font-medium whitespace-nowrap">
                {index + 1}. {step.title.split(' ')[0]}
              </span>
            </button>
            {index < steps.length - 1 && (
              <ChevronRight className="w-5 h-5 text-gray-400 mx-2" />
            )}
          </div>
        ))}
      </div>

      {/* Step Display */}
      <div className="space-y-4">
        <div className="border-b border-gray-200 pb-4">
          <h3 className="text-xl font-semibold text-gray-900">
            {steps[selectedStep].title}
          </h3>
          <p className="mt-1 text-sm text-gray-600">
            {steps[selectedStep].description}
          </p>
        </div>

        {/* Image Display */}
        <div className="relative bg-gray-50 rounded-lg p-4">
          <img
            src={`http://localhost:8000/api/result/${requestId}/${steps[selectedStep].name}`}
            alt={steps[selectedStep].title}
            className="w-full max-w-2xl mx-auto rounded-lg shadow-md"
            onError={(e) => {
              e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="400" height="300"%3E%3Crect fill="%23f0f0f0" width="400" height="300"/%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" fill="%23999" font-size="18"%3EImage loading...%3C/text%3E%3C/svg%3E';
            }}
          />
        </div>

        {/* Navigation Buttons */}
        <div className="flex justify-between pt-4">
          <button
            onClick={() => setSelectedStep(Math.max(0, selectedStep - 1))}
            disabled={selectedStep === 0}
            className={`
              px-4 py-2 rounded-lg font-medium transition-colors
              ${selectedStep === 0
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}
            `}
          >
            Previous Step
          </button>
          <button
            onClick={() => setSelectedStep(Math.min(steps.length - 1, selectedStep + 1))}
            disabled={selectedStep === steps.length - 1}
            className={`
              px-4 py-2 rounded-lg font-medium transition-colors
              ${selectedStep === steps.length - 1
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-indigo-600 text-white hover:bg-indigo-700'}
            `}
          >
            Next Step
          </button>
        </div>
      </div>
    </div>
  );
}

