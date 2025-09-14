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
      <div className="relative z-10 flex flex-col items-center justify-center h-full px-4 sm:px-8 -mt-8" style={{ zIndex: 1 }}>
        <div className="w-full max-w-4xl">
          <h2 className="text-2xl sm:text-3xl font-bold text-gray-800 text-center mb-8">
            Device Configuration
          </h2>
          
          {isLoading ? (
            <div className="text-center">
              <div className="w-16 h-16 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p className="text-gray-600">Detecting devices...</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Audience Camera */}
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-bold text-gray-800">Audience Camera</h3>
                </div>
                <select
                  value={selectedAudienceCamera}
                  onChange={(e) => setSelectedAudienceCamera(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
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
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-bold text-gray-800">Artist Camera</h3>
                </div>
                <select
                  value={selectedArtistCamera}
                  onChange={(e) => setSelectedArtistCamera(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
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
              <div className="space-y-3">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-bold text-gray-800">Artist Microphone</h3>
                </div>
                <select
                  value={selectedArtistMic}
                  onChange={(e) => setSelectedArtistMic(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
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

          {/* Device Status */}
          <div className="mt-8 bg-gray-50 rounded-xl p-4">
            <h4 className="text-lg font-semibold text-gray-800 mb-4 text-center">Device Status</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Audience Camera:</span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  selectedAudienceCamera ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                }`}>
                  {selectedAudienceCamera ? 'Connected' : 'Not Selected'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Artist Camera:</span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  selectedArtistCamera ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                }`}>
                  {selectedArtistCamera ? 'Connected' : 'Not Selected'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Artist Microphone:</span>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                  selectedArtistMic ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                }`}>
                  {selectedArtistMic ? 'Connected' : 'Not Selected'}
                </span>
              </div>
            </div>
          </div>

          {/* Continue Button */}
          {canContinue && (
            <div className="text-center mt-8">
              <button
                onClick={handleContinue}
                className="bg-gray-800 text-white px-8 py-3 rounded-full font-semibold text-lg hover:bg-gray-700 transition-all duration-300"
              >
                Continue to Live Interface
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default DeviceSettingsPage
