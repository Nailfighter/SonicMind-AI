import React, { useState, useRef } from 'react'

const UploadVideoPage = ({ profile, onBack, onContinue }) => {
  const [uploadedFile, setUploadedFile] = useState(null)
  const [selectedInstrument, setSelectedInstrument] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const instruments = [
    { id: 'guitar', name: 'Guitar', icon: '🎸' },
    { id: 'piano', name: 'Piano', icon: '🎹' },
    { id: 'drums', name: 'Drums', icon: '🥁' },
    { id: 'bass', name: 'Bass', icon: '🎸' },
    { id: 'vocals', name: 'Vocals', icon: '🎤' },
    { id: 'violin', name: 'Violin', icon: '🎻' },
    { id: 'saxophone', name: 'Saxophone', icon: '🎷' },
    { id: 'trumpet', name: 'Trumpet', icon: '🎺' },
    { id: 'other', name: 'Other', icon: '🎵' }
  ]

  const handleFileUpload = (file) => {
    if (file && file.type.startsWith('video/')) {
      setUploadedFile(file)
    } else {
      alert('Please upload a valid video file')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleFileInputChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      handleFileUpload(file)
    }
  }

  const handleInstrumentSelect = (instrument) => {
    setSelectedInstrument(instrument)
  }

  const handleContinue = () => {
    if (uploadedFile && selectedInstrument) {
      const uploadData = {
        file: uploadedFile,
        instrument: selectedInstrument
      }
      console.log('Processing video:', uploadedFile.name, 'for instrument:', selectedInstrument.name)
      onContinue(uploadData)
    }
  }

  return (
    <div className="w-screen h-screen bg-gradient-to-br from-gray-50 to-gray-100 relative overflow-hidden">
      {/* Grid overlay */}
      <div 
        className="absolute inset-0 opacity-15"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px)
          `,
          backgroundSize: '20px 20px'
        }}
      />
      
      {/* Back button - positioned absolutely */}
      <button 
        onClick={() => {
          console.log('Back button clicked!')
          onBack()
        }}
        className="absolute top-6 left-6 z-50 flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors duration-200 cursor-pointer p-2 rounded"
        style={{ zIndex: 9999 }}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        <span className="font-medium">Back</span>
      </button>
      
      {/* Header */}
      <div className="relative z-10 flex justify-end items-center p-6">
        {/* Profile info */}
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-800 via-gray-700 to-gray-900 flex items-center justify-center shadow-lg">
            <span className="text-lg font-bold text-white">
              {profile.name.charAt(0).toUpperCase()}
            </span>
          </div>
          <div className="text-right">
            <h3 className="text-lg font-bold text-gray-800">{profile.name}</h3>
            <p className="text-sm text-gray-600">Pro Mode</p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full px-4 sm:px-8 -mt-16" style={{ zIndex: 1 }}>
        <div className="w-full max-w-2xl">
          {/* File Upload Section */}
          <div className="mb-12">
            <h2 className="text-2xl sm:text-3xl font-bold text-gray-800 text-center mb-8">
              Upload Your Video
            </h2>
            
            <div
              className={`border-2 border-dashed rounded-2xl p-6 sm:p-8 text-center transition-all duration-300 ${
                isDragOver
                  ? 'border-cyan-500 bg-cyan-50'
                  : uploadedFile
                  ? 'border-green-500 bg-green-50'
                  : 'border-gray-300 bg-white hover:border-gray-400'
              }`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleFileInputChange}
                className="hidden"
              />
              
              {uploadedFile ? (
                <div className="space-y-4">
                  <div className="w-16 h-16 bg-green-500 rounded-full flex items-center justify-center mx-auto">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-800 mb-2">File Uploaded Successfully!</h3>
                    <p className="text-gray-600 mb-2">{uploadedFile.name}</p>
                    <p className="text-sm text-gray-500">
                      {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      setUploadedFile(null)
                    }}
                    className="text-sm text-gray-500 hover:text-gray-700 underline"
                  >
                    Remove file
                  </button>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="w-16 h-16 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center mx-auto">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-800 mb-2">
                      {isDragOver ? 'Drop your video here' : 'Upload Video File'}
                    </h3>
                    <p className="text-gray-600 mb-4">
                      Drag and drop your video file here, or click to browse
                    </p>
                    <p className="text-sm text-gray-500">
                      Supports MP4, MOV, AVI, and other video formats
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Instrument Selection */}
          {uploadedFile && (
            <div className="mb-8">
              <h2 className="text-2xl sm:text-3xl font-bold text-gray-800 text-center mb-8">
                Select Your Instrument
              </h2>
              
              <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-5 gap-4">
                {instruments.map((instrument) => (
                  <div
                    key={instrument.id}
                    className={`p-4 rounded-xl border-2 cursor-pointer transition-all duration-300 ${
                      selectedInstrument?.id === instrument.id
                        ? 'border-purple-500 bg-gradient-to-br from-purple-50 to-pink-50 shadow-lg'
                        : 'border-gray-200 bg-white hover:border-purple-300 hover:shadow-md'
                    }`}
                    onClick={() => handleInstrumentSelect(instrument)}
                  >
                    <div className="text-center">
                      <div className="text-3xl sm:text-4xl mb-2">{instrument.icon}</div>
                      <p className="text-sm font-medium text-gray-800">{instrument.name}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Continue button */}
          {uploadedFile && selectedInstrument && (
            <div className="text-center">
              <button
                onClick={handleContinue}
                className="bg-gray-800 text-white px-8 py-3 rounded-full font-semibold text-lg hover:bg-gray-700 transition-all duration-300"
              >
                Process Video for {selectedInstrument.name}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default UploadVideoPage
