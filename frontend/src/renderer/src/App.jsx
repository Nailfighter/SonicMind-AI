import React, { useState } from 'react'
import './assets/main.css'
export default function App() {
  const [bands, setBands] = useState({
    bass: 50,
    mid: 50,
    treble: 50
  })

  const [backendData, setBackendData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [currentRoute, setCurrentRoute] = useState('time')
  const [audioFile, setAudioFile] = useState(null)
  const [audioData, setAudioData] = useState(null)
  const [audioLoading, setAudioLoading] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)

  const handleChange = (e) => {
    setBands({
      ...bands,
      [e.target.name]: e.target.value
    })
  }

  const handleGetBackendData = async (route = 'time') => {
    setLoading(true)
    setCurrentRoute(route)
    try {
      const data = await window.api.getBackendData(route)
      setBackendData(data)
    } catch (error) {
      console.error('Error fetching backend data:', error)
      setBackendData({ error: error.message })
    } finally {
      setLoading(false)
    }
  }

  const handleAudioUpload = (event) => {
    const file = event.target.files[0]
    if (file && file.type.startsWith('audio/')) {
      // Check file size (limit to 50MB)
      const maxSize = 50 * 1024 * 1024 // 50MB
      if (file.size > maxSize) {
        alert('File size too large. Please select a file smaller than 50MB.')
        return
      }

      setAudioFile(file)
      const url = URL.createObjectURL(file)
      setAudioUrl(url)
      setAudioData(null) // Clear previous data
    } else {
      alert('Please select a valid audio file')
    }
  }

  const handleAudioProcess = async (processType) => {
    if (!audioFile) {
      alert('Please upload an audio file first')
      return
    }

    setAudioLoading(true)
    try {
      // Convert file to base64 using FileReader (more efficient for large files)
      const base64 = await new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
          // Remove the data URL prefix (data:audio/wav;base64,)
          const base64String = reader.result.split(',')[1]
          resolve(base64String)
        }
        reader.onerror = reject
        reader.readAsDataURL(audioFile)
      })

      const data = await window.api.processAudio(base64, audioFile.name, processType)
      setAudioData(data)
    } catch (error) {
      console.error('Error processing audio:', error)
      setAudioData({ error: error.message })
    } finally {
      setAudioLoading(false)
    }
  }

  const availableRoutes = [
    { key: 'time', label: 'Time Data', description: 'Get current time and timestamp' },
    { key: 'random', label: 'Random Number', description: 'Generate a random number 1-100' },
    { key: 'system', label: 'System Info', description: 'Get system platform and Python version' },
    { key: 'weather', label: 'Weather Data', description: 'Get mock weather information' }
  ]

  const audioProcessTypes = [
    { key: 'analyze', label: 'Analyze Audio', description: 'Get audio properties and metadata' },
    { key: 'convert', label: 'Convert Format', description: 'Convert audio to different format' },
    {
      key: 'extract',
      label: 'Extract Features',
      description: 'Extract audio features and frequencies'
    },
    { key: 'trim', label: 'Trim Audio', description: 'Trim audio to specific duration' }
  ]

  return (
    <div className="h-screen w-screen flex flex-col items-center justify-center bg-gray-900 text-white font-sans">
      <h1 className="text-3xl font-bold mb-8">🎵 Music Equalizer Dashboard</h1>

      {/* Backend Data Section */}
      <div className="mb-8 p-6 bg-gray-800 rounded-lg border border-gray-700">
        <h2 className="text-xl font-semibold mb-4 text-center">Backend API Routes</h2>

        {/* Route Selection Buttons */}
        <div className="grid grid-cols-2 gap-3 mb-6">
          {availableRoutes.map((route) => (
            <button
              key={route.key}
              onClick={() => handleGetBackendData(route.key)}
              disabled={loading}
              className={`px-4 py-3 rounded-lg font-medium transition-colors duration-200 ${
                currentRoute === route.key
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white'
              }`}
            >
              {loading && currentRoute === route.key ? 'Loading...' : route.label}
            </button>
          ))}
        </div>

        {/* Route Descriptions */}
        <div className="text-sm text-gray-300 mb-4">
          <p className="text-center">
            Selected:{' '}
            <span className="font-medium text-green-400">
              {availableRoutes.find((r) => r.key === currentRoute)?.label}
            </span>
          </p>
          <p className="text-center mt-1">
            {availableRoutes.find((r) => r.key === currentRoute)?.description}
          </p>
        </div>

        {backendData && (
          <div className="mt-4 p-4 bg-gray-700 rounded-lg">
            <h3 className="text-lg font-medium mb-2">
              Backend Response ({backendData.function || 'Unknown'}):
            </h3>
            {backendData.error ? (
              <div className="space-y-2">
                <p className="text-red-400">Error: {backendData.error}</p>
                {backendData.available_routes && (
                  <div>
                    <p className="text-yellow-400">Available routes:</p>
                    <ul className="list-disc list-inside text-yellow-300">
                      {backendData.available_routes.map((route) => (
                        <li key={route}>{route}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-2">
                {/* Dynamic display based on function type */}
                {backendData.function === 'get_time_data' && (
                  <>
                    <p>
                      <span className="font-medium">Time:</span> {backendData.time}
                    </p>
                    <p>
                      <span className="font-medium">Timestamp:</span> {backendData.timestamp}
                    </p>
                  </>
                )}
                {backendData.function === 'get_random_number' && (
                  <>
                    <p>
                      <span className="font-medium">Random Number:</span>{' '}
                      {backendData.random_number}
                    </p>
                    <p>
                      <span className="font-medium">Range:</span> {backendData.range}
                    </p>
                  </>
                )}
                {backendData.function === 'get_system_info' && (
                  <>
                    <p>
                      <span className="font-medium">Platform:</span> {backendData.platform}
                    </p>
                    <p>
                      <span className="font-medium">Python Version:</span>{' '}
                      {backendData.python_version}
                    </p>
                  </>
                )}
                {backendData.function === 'get_weather_data' && (
                  <>
                    <p>
                      <span className="font-medium">Temperature:</span> {backendData.temperature}°C
                    </p>
                    <p>
                      <span className="font-medium">Condition:</span> {backendData.condition}
                    </p>
                    <p>
                      <span className="font-medium">Humidity:</span> {backendData.humidity}%
                    </p>
                  </>
                )}
                <p>
                  <span className="font-medium">Message:</span> {backendData.message}
                </p>
                <p>
                  <span className="font-medium">Timestamp:</span> {backendData.timestamp}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Audio Processing Section */}
      <div className="mb-8 p-6 bg-gray-800 rounded-lg border border-gray-700 w-full max-w-4xl">
        <h2 className="text-xl font-semibold mb-4 text-center">🎵 Audio Processing</h2>

        {/* File Upload */}
        <div className="mb-6">
          <label className="block text-sm font-medium mb-2">Upload Audio File:</label>
          <input
            type="file"
            accept="audio/*"
            onChange={handleAudioUpload}
            className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
          />
          {audioFile && (
            <div className="mt-2 p-3 bg-gray-700 rounded-lg">
              <p className="text-sm">
                <span className="font-medium">File:</span> {audioFile.name}
                <span className="ml-2 text-gray-400">
                  ({(audioFile.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </p>
              {audioUrl && (
                <audio controls className="w-full mt-2" src={audioUrl}>
                  Your browser does not support the audio element.
                </audio>
              )}
            </div>
          )}
        </div>

        {/* Audio Processing Buttons */}
        {audioFile && (
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-3">Process Audio:</h3>
            <div className="grid grid-cols-2 gap-3">
              {audioProcessTypes.map((process) => (
                <button
                  key={process.key}
                  onClick={() => handleAudioProcess(process.key)}
                  disabled={audioLoading}
                  className="px-4 py-3 bg-purple-600 hover:bg-purple-700 disabled:bg-purple-400 text-white font-medium rounded-lg transition-colors duration-200"
                >
                  {audioLoading ? 'Processing...' : process.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Audio Processing Results */}
        {audioData && (
          <div className="mt-4 p-4 bg-gray-700 rounded-lg">
            <h3 className="text-lg font-medium mb-2">
              Audio Processing Result ({audioData.function || 'Unknown'}):
            </h3>
            {audioData.error ? (
              <p className="text-red-400">Error: {audioData.error}</p>
            ) : (
              <div className="space-y-2">
                {audioData.function === 'analyze_audio' && (
                  <>
                    <p>
                      <span className="font-medium">Duration:</span> {audioData.duration}s
                    </p>
                    <p>
                      <span className="font-medium">Sample Rate:</span> {audioData.sample_rate} Hz
                    </p>
                    <p>
                      <span className="font-medium">Channels:</span> {audioData.channels}
                    </p>
                    <p>
                      <span className="font-medium">Format:</span> {audioData.format}
                    </p>
                    <p>
                      <span className="font-medium">Bit Rate:</span> {audioData.bit_rate} kbps
                    </p>
                  </>
                )}
                {audioData.function === 'convert_audio' && (
                  <>
                    <p>
                      <span className="font-medium">Original Format:</span>{' '}
                      {audioData.original_format}
                    </p>
                    <p>
                      <span className="font-medium">Converted Format:</span>{' '}
                      {audioData.converted_format}
                    </p>
                    <p>
                      <span className="font-medium">File Size:</span> {audioData.file_size} bytes
                    </p>
                  </>
                )}
                {audioData.function === 'extract_features' && (
                  <>
                    <p>
                      <span className="font-medium">RMS Energy:</span> {audioData.rms_energy}
                    </p>
                    <p>
                      <span className="font-medium">Spectral Centroid:</span>{' '}
                      {audioData.spectral_centroid} Hz
                    </p>
                    <p>
                      <span className="font-medium">Zero Crossing Rate:</span> {audioData.zcr}
                    </p>
                    <p>
                      <span className="font-medium">MFCC Features:</span>{' '}
                      {audioData.mfcc_features?.length || 0} coefficients
                    </p>
                  </>
                )}
                {audioData.function === 'trim_audio' && (
                  <>
                    <p>
                      <span className="font-medium">Original Duration:</span>{' '}
                      {audioData.original_duration}s
                    </p>
                    <p>
                      <span className="font-medium">Trimmed Duration:</span>{' '}
                      {audioData.trimmed_duration}s
                    </p>
                    <p>
                      <span className="font-medium">Start Time:</span> {audioData.start_time}s
                    </p>
                    <p>
                      <span className="font-medium">End Time:</span> {audioData.end_time}s
                    </p>
                  </>
                )}
                <p>
                  <span className="font-medium">Message:</span> {audioData.message}
                </p>
                <p>
                  <span className="font-medium">Timestamp:</span> {audioData.timestamp}
                </p>
              </div>
            )}
          </div>
        )}
      </div>

    </div>
  )
}
