import React, { useState, useEffect, useRef } from 'react'
import { useDeviceManager, useDeviceSelection, useCameraPreview } from '../hooks/useDeviceManager'

/**
 * Enhanced Device Settings Page with Backend Integration
 * 
 * This component provides a comprehensive device selection interface that
 * integrates with both web API devices and backend audio devices.
 */
const EnhancedDeviceSettings = ({ profile, onBack, onContinue }) => {
  const {
    devices,
    status,
    hasRequiredDevices,
    refreshDevices,
    selectDevices,
    getRecommendedDevices,
    validateSelection
  } = useDeviceManager()
  
  const { selection, updateDevice, updateMultipleDevices, isComplete } = useDeviceSelection()
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [testResults, setTestResults] = useState(null)
  const [isTestingDevices, setIsTestingDevices] = useState(false)
  
  // Camera preview refs
  const audienceCameraRef = useRef(null)
  const artistCameraRef = useRef(null)
  
  // Camera previews
  const audienceCameraPreview = useCameraPreview()
  const artistCameraPreview = useCameraPreview()

  // Auto-apply recommended devices when devices are loaded
  useEffect(() => {
    if (status.isInitialized && devices.cameras.length > 0) {
      const recommendations = getRecommendedDevices()
      console.log('📋 Applying recommended devices:', recommendations)
      
      updateMultipleDevices({
        audienceCamera: recommendations.audienceCamera,
        artistCamera: recommendations.artistCamera,
        artistMicrophone: recommendations.artistMicrophone,
        audioInput: recommendations.audioInput,
        audioOutput: recommendations.audioOutput
      })
    }
  }, [status.isInitialized, devices, getRecommendedDevices, updateMultipleDevices])

  // Update camera previews when selection changes
  useEffect(() => {
    if (selection.audienceCamera && audienceCameraRef.current) {
      audienceCameraPreview.startPreview(selection.audienceCamera).then(() => {
        if (audienceCameraPreview.stream && audienceCameraRef.current) {
          audienceCameraRef.current.srcObject = audienceCameraPreview.stream
        }
      })
    }
  }, [selection.audienceCamera, audienceCameraPreview])

  useEffect(() => {
    if (selection.artistCamera && artistCameraRef.current) {
      artistCameraPreview.startPreview(selection.artistCamera).then(() => {
        if (artistCameraPreview.stream && artistCameraRef.current) {
          artistCameraRef.current.srcObject = artistCameraPreview.stream
        }
      })
    }
  }, [selection.artistCamera, artistCameraPreview])

  const handleDeviceChange = (deviceType, deviceValue) => {
    const deviceId = deviceValue === '' ? null : deviceValue
    updateDevice(deviceType, deviceId)
  }

  const handleTestDevices = async () => {
    setIsTestingDevices(true)
    setTestResults(null)
    
    try {
      const success = await selectDevices(selection)
      setTestResults({
        success,
        message: success ? 'All devices tested successfully!' : 'Some devices failed testing',
        timestamp: new Date().toLocaleTimeString()
      })
    } catch (error) {
      setTestResults({
        success: false,
        message: error.message,
        timestamp: new Date().toLocaleTimeString()
      })
    } finally {
      setIsTestingDevices(false)
    }
  }

  const handleContinue = async () => {
    const validation = validateSelection(selection)
    
    if (!validation.valid) {
      setTestResults({
        success: false,
        message: validation.errors.join(', '),
        timestamp: new Date().toLocaleTimeString()
      })
      return
    }
    
    // Test devices before continuing
    setIsTestingDevices(true)
    const success = await selectDevices(selection)
    setIsTestingDevices(false)
    
    if (success) {
      // Convert selection to legacy format for compatibility
      const deviceSettings = {
        AudienceCamera: selection.audienceCamera,
        ArtistCamera: selection.artistCamera,
        ArtistMicrophone: selection.artistMicrophone,
        AudioInput: selection.audioInput,
        AudioOutput: selection.audioOutput
      }
      
      onContinue(deviceSettings)
    } else {
      setTestResults({
        success: false,
        message: 'Device testing failed. Please check your device connections.',
        timestamp: new Date().toLocaleTimeString()
      })
    }
  }

  const canContinue = isComplete(['audioInput', 'audioOutput']) && hasRequiredDevices
  const hasOptionalDevices = selection.audienceCamera || selection.artistCamera || selection.artistMicrophone

  if (status.isLoading) {
    return (
      <div className="w-screen h-screen bg-gradient-to-br from-gray-50 to-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-20 h-20 border-4 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-lg text-gray-600">Discovering devices...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a few seconds</p>
        </div>
      </div>
    )
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

      {/* Back button */}
      <button
        onClick={onBack}
        className="absolute top-6 left-6 z-50 flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors duration-200 cursor-pointer p-2 rounded"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
        </svg>
        <span className="font-medium">Back</span>
      </button>

      {/* Header */}
      <div className="relative z-10 flex justify-end items-center p-6">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-gray-800 via-gray-700 to-gray-900 flex items-center justify-center shadow-lg">
            <span className="text-lg font-bold text-white">
              {profile.name.charAt(0).toUpperCase()}
            </span>
          </div>
          <div className="text-right">
            <h3 className="text-lg font-bold text-gray-800">{profile.name}</h3>
            <p className="text-sm text-gray-600">Enhanced Audio Setup</p>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="relative z-10 flex flex-col items-center justify-start h-full px-4 sm:px-6 lg:px-8 pt-12 pb-8 overflow-y-auto">
        <div className="w-full max-w-4xl">
          <h2 className="text-xl sm:text-2xl font-bold text-gray-800 text-center mb-3 sm:mb-4">
            Device Configuration
          </h2>
          <p className="text-sm text-gray-600 text-center mb-6">
            Configure your audio and video devices for optimal performance
          </p>

          {/* Error display */}
          {status.error && (
            <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg">
              <div className="text-sm font-medium text-red-800">Configuration Error</div>
              <div className="text-sm text-red-600 mt-1">{status.error}</div>
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
            {/* Audio Devices (Required) */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                <span className="w-8 h-8 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center mr-3">
                  <span className="text-white text-sm">🎧</span>
                </span>
                Audio Devices (Required)
              </h3>

              {/* Audio Input */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-4">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-600 rounded-full flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-lg font-bold text-gray-800">Audio Input</h4>
                    <p className="text-sm text-gray-600">Professional audio interface or microphone</p>
                  </div>
                </div>
                <select
                  value={selection.audioInput || ''}
                  onChange={(e) => handleDeviceChange('audioInput', e.target.value)}
                  className="w-full p-3 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="">Choose audio input...</option>
                  {devices.audioInputs.map((device) => (
                    <option key={device.index} value={device.index}>
                      {device.name} ({device.channels} channels)
                    </option>
                  ))}
                </select>
              </div>

              {/* Audio Output */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-4">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-600 rounded-full flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.636 15.364a5 5 0 010-7.072m-2.828-9.9a9 9 0 000 12.728" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-lg font-bold text-gray-800">Audio Output</h4>
                    <p className="text-sm text-gray-600">Studio monitors or headphones</p>
                  </div>
                </div>
                <select
                  value={selection.audioOutput || ''}
                  onChange={(e) => handleDeviceChange('audioOutput', e.target.value)}
                  className="w-full p-3 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="">Choose audio output...</option>
                  {devices.audioOutputs.map((device) => (
                    <option key={device.index} value={device.index}>
                      {device.name} ({device.channels} channels)
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Camera Devices (Optional) */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-800 flex items-center">
                <span className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center mr-3">
                  <span className="text-white text-sm">📷</span>
                </span>
                Camera Devices (Optional)
              </h3>

              {/* Audience Camera */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-4">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <h4 className="text-lg font-bold text-gray-800">Audience Camera</h4>
                    <p className="text-sm text-gray-600">Camera for audience detection</p>
                  </div>
                </div>
                <select
                  value={selection.audienceCamera || ''}
                  onChange={(e) => handleDeviceChange('audienceCamera', e.target.value)}
                  className="w-full p-3 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-cyan-500 focus:border-transparent transition-all duration-200 mb-3"
                >
                  <option value="">Choose camera...</option>
                  {devices.cameras.map((camera) => (
                    <option key={camera.id} value={camera.id}>
                      {camera.name}
                    </option>
                  ))}
                </select>
                {selection.audienceCamera && (
                  <div className="bg-gray-100 rounded-lg p-2">
                    <video
                      ref={audienceCameraRef}
                      autoPlay
                      muted
                      className="w-full h-24 object-cover rounded"
                    />
                  </div>
                )}
              </div>

              {/* Artist Camera */}
              <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-4">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 002 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-lg font-bold text-gray-800">Artist Camera</h4>
                    <p className="text-sm text-gray-600">Camera for instrument detection</p>
                  </div>
                </div>
                <select
                  value={selection.artistCamera || ''}
                  onChange={(e) => handleDeviceChange('artistCamera', e.target.value)}
                  className="w-full p-3 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all duration-200 mb-3"
                >
                  <option value="">Choose camera...</option>
                  {devices.cameras.map((camera) => (
                    <option key={camera.id} value={camera.id}>
                      {camera.name}
                    </option>
                  ))}
                </select>
                {selection.artistCamera && (
                  <div className="bg-gray-100 rounded-lg p-2">
                    <video
                      ref={artistCameraRef}
                      autoPlay
                      muted
                      className="w-full h-24 object-cover rounded"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Advanced Options Toggle */}
          <div className="text-center mb-4">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-sm text-cyan-600 hover:text-cyan-800 transition-colors"
            >
              {showAdvanced ? '▼ Hide' : '▶ Show'} Advanced Options
            </button>
          </div>

          {/* Advanced Options */}
          {showAdvanced && (
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Advanced Settings</h3>
              
              {/* Artist Microphone */}
              <div className="bg-white rounded-lg border border-gray-200 p-4">
                <div className="flex items-center space-x-3 mb-3">
                  <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-red-600 rounded-full flex items-center justify-center">
                    <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="text-lg font-bold text-gray-800">Artist Microphone</h4>
                    <p className="text-sm text-gray-600">Additional microphone for detection</p>
                  </div>
                </div>
                <select
                  value={selection.artistMicrophone || ''}
                  onChange={(e) => handleDeviceChange('artistMicrophone', e.target.value)}
                  className="w-full p-3 text-sm border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="">Choose microphone...</option>
                  {devices.microphones.map((mic) => (
                    <option key={mic.id} value={mic.id}>
                      {mic.name}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          )}

          {/* Device Status */}
          <div className="mb-6 p-4 bg-white rounded-lg shadow-lg border border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-800">Device Status</h3>
              <button
                onClick={() => refreshDevices(true)}
                disabled={status.isRefreshing}
                className="bg-gray-500 hover:bg-gray-600 disabled:bg-gray-300 text-white px-3 py-1 rounded text-sm transition-colors"
              >
                {status.isRefreshing ? '🔄' : '↻'} Refresh
              </button>
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Audio In:</span>
                <span className={`text-sm font-medium ${selection.audioInput ? 'text-green-600' : 'text-red-600'}`}>
                  {selection.audioInput ? '✓' : '✗'}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Audio Out:</span>
                <span className={`text-sm font-medium ${selection.audioOutput ? 'text-green-600' : 'text-red-600'}`}>
                  {selection.audioOutput ? '✓' : '✗'}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Audience Cam:</span>
                <span className={`text-sm font-medium ${selection.audienceCamera ? 'text-green-600' : 'text-gray-400'}`}>
                  {selection.audienceCamera ? '✓' : '-'}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Artist Cam:</span>
                <span className={`text-sm font-medium ${selection.artistCamera ? 'text-green-600' : 'text-gray-400'}`}>
                  {selection.artistCamera ? '✓' : '-'}
                </span>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <span className="text-sm text-gray-600">Artist Mic:</span>
                <span className={`text-sm font-medium ${selection.artistMicrophone ? 'text-green-600' : 'text-gray-400'}`}>
                  {selection.artistMicrophone ? '✓' : '-'}
                </span>
              </div>
            </div>
          </div>

          {/* Test Results */}
          {testResults && (
            <div className={`mb-6 p-4 rounded-lg border ${testResults.success ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
              <div className="flex items-center justify-between">
                <div className={`text-sm font-medium ${testResults.success ? 'text-green-800' : 'text-red-800'}`}>
                  {testResults.success ? '✅ Device Test Results' : '❌ Device Test Results'}
                </div>
                <div className="text-xs text-gray-500">{testResults.timestamp}</div>
              </div>
              <div className={`text-sm mt-1 ${testResults.success ? 'text-green-600' : 'text-red-600'}`}>
                {testResults.message}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-center space-x-4">
            <button
              onClick={handleTestDevices}
              disabled={!canContinue || isTestingDevices}
              className="bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-300 text-white px-6 py-3 rounded-lg font-semibold text-sm transition-colors"
            >
              {isTestingDevices ? '🔄 Testing...' : '🧪 Test Devices'}
            </button>
            
            <button
              onClick={handleContinue}
              disabled={!canContinue || isTestingDevices}
              className="bg-gray-800 hover:bg-gray-700 disabled:bg-gray-300 text-white px-8 py-3 rounded-lg font-semibold text-lg transition-colors"
            >
              {isTestingDevices ? 'Testing...' : 'Continue to Live Interface'}
            </button>
          </div>

          {!canContinue && (
            <div className="text-center mt-4">
              <p className="text-sm text-gray-500">
                Please select audio input and output devices to continue
              </p>
              {!hasRequiredDevices && (
                <p className="text-sm text-red-600 mt-1">
                  ⚠️ No audio devices found. Please check your connections and refresh.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default EnhancedDeviceSettings