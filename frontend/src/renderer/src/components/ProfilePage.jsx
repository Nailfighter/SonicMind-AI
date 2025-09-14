import React, { useState } from 'react'

const ProfilePage = ({ profile, onBack, onModeSelect }) => {
  const [selectedMode, setSelectedMode] = useState(null)

  const handleModeSelect = (mode) => {
    setSelectedMode(mode)
    console.log('Selected mode:', mode)
    if (onModeSelect) {
      onModeSelect(mode)
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

      {/* Header */}
      <div className="relative z-10 flex justify-start items-center p-6">
        {/* Back button */}
        <button
          onClick={onBack}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors duration-200 cursor-pointer"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
          <span className="font-medium">Back</span>
        </button>
      </div>

      {/* Main content - moved up more */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full px-4 sm:px-8 -mt-16 sm:-mt-24">
        {/* Mode selection - perfectly centered */}
        <div className="w-full max-w-4xl flex flex-col items-center justify-center">
          <h2 className="text-3xl sm:text-4xl font-bold text-gray-800 text-center mb-8 sm:mb-12">
            Choose Your Workflow
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 w-full max-w-3xl">
            {/* Live Setting Mode */}
            <div
              className={`p-6 sm:p-8 rounded-2xl border-2 cursor-pointer transition-all duration-300 ${
                selectedMode === 'live'
                  ? 'border-cyan-500 bg-gradient-to-br from-cyan-50 to-blue-50 shadow-xl'
                  : 'border-gray-200 bg-white hover:border-cyan-300 hover:shadow-lg'
              }`}
              onClick={() => handleModeSelect('live')}
            >
              <div className="text-center">
                <div className="w-12 h-12 sm:w-16 sm:h-16 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-3 sm:mb-4 shadow-lg">
                  <svg
                    className="w-6 h-6 sm:w-8 sm:h-8 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg sm:text-xl font-bold text-gray-800 mb-2">Live Setting</h3>
                <p className="text-sm sm:text-base text-gray-600">
                  Real-time audio processing and monitoring for live performances and recording
                  sessions
                </p>
              </div>
            </div>

            {/* Upload Video Mode */}
            <div
              className={`p-6 sm:p-8 rounded-2xl border-2 cursor-pointer transition-all duration-300 ${
                selectedMode === 'upload'
                  ? 'border-purple-500 bg-gradient-to-br from-purple-50 to-pink-50 shadow-xl'
                  : 'border-gray-200 bg-white hover:border-purple-300 hover:shadow-lg'
              }`}
              onClick={() => handleModeSelect('upload')}
            >
              <div className="text-center">
                <div className="w-12 h-12 sm:w-16 sm:h-16 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center mx-auto mb-3 sm:mb-4 shadow-lg">
                  <svg
                    className="w-6 h-6 sm:w-8 sm:h-8 text-white"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                </div>
                <h3 className="text-lg sm:text-xl font-bold text-gray-800 mb-2">Upload Video</h3>
                <p className="text-sm sm:text-base text-gray-600">
                  Upload and process video files for post-production audio enhancement and analysis
                </p>
              </div>
            </div>
          </div>

          {/* Continue button */}
          {selectedMode && (
            <div className="text-center mt-6 sm:mt-8">
              <button
                onClick={() => handleModeSelect(selectedMode)}
                className="bg-gray-800 text-white px-6 sm:px-8 py-2.5 sm:py-3 rounded-full font-semibold text-base sm:text-lg hover:bg-gray-700 transition-all duration-300"
              >
                Continue to {selectedMode === 'live' ? 'Live Setting' : 'Upload Video'}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ProfilePage
