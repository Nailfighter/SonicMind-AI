import React, { useState, useEffect } from 'react'

const DeviceSettingsPage = ({ profile, onBack, onContinue }) => {
  const [audienceCameras, setAudienceCameras] = useState([])
  const [artistCameras, setArtistCameras] = useState([])
  const [artistMicrophones, setArtistMicrophones] = useState([])
  const [selectedAudienceCamera, setSelectedAudienceCamera] = useState('')
  const [selectedArtistCamera, setSelectedArtistCamera] = useState('')
  const [selectedArtistMic, setSelectedArtistMic] = useState('')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadDevices = async () => {
      setIsLoading(true)
      
      try {
        // Get real devices from the system
        const devices = await navigator.mediaDevices.enumerateDevices()
        
        const cameras = devices.filter(device => device.kind === 'videoinput')
        const microphones = devices.filter(device => device.kind === 'audioinput')
        
        // For demo purposes, we'll use the same camera list for both audience and artist
        // In a real app, you might want to differentiate or allow the same camera for both
        setAudienceCameras(cameras.map((device, index) => ({
          id: device.deviceId,
          name: device.label || `Camera ${index + 1}`,
          type: 'camera'
        })))
        
        setArtistCameras(cameras.map((device, index) => ({
          id: device.deviceId,
          name: device.label || `Camera ${index + 1}`,
          type: 'camera'
        })))
        
        setArtistMicrophones(microphones.map((device, index) => ({
          id: device.deviceId,
          name: device.label || `Microphone ${index + 1}`,
          type: 'microphone'
        })))
        
        setIsLoading(false)
      } catch (error) {
        console.error('Error accessing media devices:', error)
        // Fallback to empty arrays if permission denied
        setAudienceCameras([])
        setArtistCameras([])
        setArtistMicrophones([])
        setIsLoading(false)
      }
    }
    
    loadDevices()
  }, [])

  const handleContinue = () => {
    if (selectedAudienceCamera && selectedArtistCamera && selectedArtistMic) {
      console.log('Device settings completed:', {
        audienceCamera: selectedAudienceCamera,
        artistCamera: selectedArtistCamera,
        artistMic: selectedArtistMic
      })
      onContinue()
    }
  }

  const canContinue = selectedAudienceCamera && selectedArtistCamera && selectedArtistMic

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
        onClick={onBack}
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
      <div className="relative z-10 flex flex-col items-center justify-start h-full px-4 sm:px-6 lg:px-8 pt-12 pb-8 overflow-y-auto" style={{ zIndex: 1 }}>
        <div className="w-full max-w-2xl">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-800 text-center mb-3 sm:mb-4">
            Device Configuration
          </h2>
          
          {isLoading ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 sm:w-20 sm:h-20 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-base sm:text-lg text-gray-600">Detecting devices...</p>
            </div>
          ) : (
            <div className="space-y-2 sm:space-y-3">
              {/* Audience Camera */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-3 sm:p-4">
                <div className="flex items-center space-x-3 sm:space-x-4 mb-2 sm:mb-3">
                  <div className="w-10 h-10 sm:w-12 sm:h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center shadow-lg">
                    <svg className="w-5 h-5 sm:w-6 sm:h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg sm:text-xl font-bold text-gray-800">Audience Camera</h3>
                    <p className="text-sm text-gray-600">Camera for audience view</p>
                  </div>
                </div>
                <select
                  value={selectedAudienceCamera}
                  onChange={(e) => setSelectedAudienceCamera(e.target.value)}
                  className="w-full p-3 sm:p-4 text-sm sm:text-base border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="">Choose camera...</option>
                  {audienceCameras.map((camera) => (
                    <option key={camera.id} value={camera.id}>
                      {camera.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Artist Camera */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-3 sm:p-4">
                <div className="flex items-center space-x-3 sm:space-x-4 mb-2 sm:mb-3">
                  <div className="w-10 h-10 sm:w-12 sm:h-12 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center shadow-lg">
                    <svg className="w-5 h-5 sm:w-6 sm:h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg sm:text-xl font-bold text-gray-800">Artist Camera</h3>
                    <p className="text-sm text-gray-600">Camera for artist view</p>
                  </div>
                </div>
                <select
                  value={selectedArtistCamera}
                  onChange={(e) => setSelectedArtistCamera(e.target.value)}
                  className="w-full p-3 sm:p-4 text-sm sm:text-base border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="">Choose camera...</option>
                  {artistCameras.map((camera) => (
                    <option key={camera.id} value={camera.id}>
                      {camera.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Artist Microphone */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-3 sm:p-4">
                <div className="flex items-center space-x-3 sm:space-x-4 mb-2 sm:mb-3">
                  <div className="w-10 h-10 sm:w-12 sm:h-12 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center shadow-lg">
                    <svg className="w-5 h-5 sm:w-6 sm:h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg sm:text-xl font-bold text-gray-800">Artist Microphone</h3>
                    <p className="text-sm text-gray-600">Microphone for audio input</p>
                  </div>
                </div>
                <select
                  value={selectedArtistMic}
                  onChange={(e) => setSelectedArtistMic(e.target.value)}
                  className="w-full p-3 sm:p-4 text-sm sm:text-base border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="">Choose microphone...</option>
                  {artistMicrophones.map((mic) => (
                    <option key={mic.id} value={mic.id}>
                      {mic.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          {/* Device Status - Compact */}
          <div className="mt-2 sm:mt-3 bg-gray-50 rounded-lg p-2 sm:p-3 shadow-md">
            <h4 className="text-sm sm:text-base font-semibold text-gray-800 mb-2 text-center">Device Status</h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <div className="flex items-center justify-between p-2 bg-white rounded-md">
                <span className="text-xs text-gray-600">Audience:</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  selectedAudienceCamera ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                }`}>
                  {selectedAudienceCamera ? '✓' : '✗'}
                </span>
              </div>
              <div className="flex items-center justify-between p-2 bg-white rounded-md">
                <span className="text-xs text-gray-600">Artist Cam:</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  selectedArtistCamera ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                }`}>
                  {selectedArtistCamera ? '✓' : '✗'}
                </span>
              </div>
              <div className="flex items-center justify-between p-2 bg-white rounded-md">
                <span className="text-xs text-gray-600">Mic:</span>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  selectedArtistMic ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                }`}>
                  {selectedArtistMic ? '✓' : '✗'}
                </span>
              </div>
            </div>
          </div>

          {/* Continue Button - Moved much higher */}
          <div className="text-center mt-1 sm:mt-2">
            {canContinue ? (
              <button
                onClick={handleContinue}
                className="bg-gray-800 text-white px-8 sm:px-12 py-3 sm:py-4 rounded-full font-semibold text-base sm:text-lg hover:bg-gray-700 transition-all duration-300 shadow-lg hover:shadow-xl"
              >
                Continue to Live Interface
              </button>
            ) : (
              <div className="text-gray-500 text-sm">
                Please select all devices to continue
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default DeviceSettingsPage
