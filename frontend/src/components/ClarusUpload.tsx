import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, CheckCircle, XCircle } from 'lucide-react';

interface ClarusUploadProps {
  onUpload: (file: File) => void;
  disabled?: boolean;
}

export default function ClarusUpload({ onUpload, disabled }: ClarusUploadProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
        setUploaded(true);
      };
      reader.readAsDataURL(file);

      // Call upload handler
      onUpload(file);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png']
    },
    maxFiles: 1,
    disabled
  });

  const handleClear = () => {
    setPreview(null);
    setUploaded(false);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-xl font-bold text-gray-900 mb-4">
        Optional: Upload Ground Truth Clarus Image
      </h3>
      <p className="text-sm text-gray-600 mb-4">
        Upload a Clarus image to compare with the enhanced result and calculate accuracy metrics.
      </p>

      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive ? 'border-green-500 bg-green-50' : 'border-gray-300 bg-gray-50'}
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-green-400 hover:bg-green-50'}
          ${uploaded ? 'border-green-500 bg-green-50' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {preview ? (
          <div className="flex flex-col items-center space-y-4">
            <CheckCircle className="w-12 h-12 text-green-600" />
            <img
              src={preview}
              alt="Clarus Preview"
              className="max-w-xs max-h-48 rounded-lg shadow-md"
            />
            <div className="text-center">
              <p className="text-lg font-medium text-green-700">
                Ground Truth Uploaded
              </p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleClear();
                }}
                className="mt-2 text-sm text-red-600 hover:text-red-700 underline"
              >
                Remove
              </button>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-4">
            <Upload className="w-12 h-12 text-gray-400" />
            <div>
              <p className="text-lg font-medium text-gray-700">
                {isDragActive ? 'Drop Clarus image here' : 'Upload Clarus Image (Optional)'}
              </p>
              <p className="mt-1 text-sm text-gray-500">
                Drag and drop or click to browse
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

