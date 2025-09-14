import React, { useState, useRef, useEffect } from 'react'
import FluidVisualizer from './FluidVisualizer'

const UploadVideoPage = ({ profile, onBack }) => {
  const [uploadedFile, setUploadedFile] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const [videoUrl, setVideoUrl] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [audioContext, setAudioContext] = useState(null)
  const [analyser, setAnalyser] = useState(null)
  const [audioSource, setAudioSource] = useState(null)
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)

  const handleFileUpload = (file) => {
    if (file && file.type.startsWith('video/')) {
      setUploadedFile(file)
      const url = URL.createObjectURL(file)
      setVideoUrl(url)
      setupAudioContext(url)
    } else {
      alert('Please upload a valid video file')
    }
  }

  const setupAudioContext = async (videoUrl) => {
    try {
      // Create audio context
      const AudioContext = window.AudioContext || window.webkitAudioContext
      const ctx = new AudioContext()
      setAudioContext(ctx)

      // Wait for the main video element to be ready
      await new Promise((resolve) => {
        if (videoRef.current) {
          if (videoRef.current.readyState >= 1) {
            resolve()
          } else {
            videoRef.current.addEventListener('loadedmetadata', resolve)
          }
        } else {
          resolve()
        }
      })

      // Create audio source from the main video element
      const source = ctx.createMediaElementSource(videoRef.current)
      setAudioSource(source)

      // Create analyser
      const analyserNode = ctx.createAnalyser()
      analyserNode.fftSize = 2048
      analyserNode.smoothingTimeConstant = 0.8
      setAnalyser(analyserNode)

      // Connect audio graph
      source.connect(analyserNode)
      analyserNode.connect(ctx.destination)
    } catch (error) {
      console.error('Error setting up audio context:', error)
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

  const handleVideoTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const handleVideoLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration)
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl)
      }
      if (audioContext) {
        audioContext.close()
      }
    }
  }, [videoUrl, audioContext])

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

      {/* Main content */}
      <div
        className="relative z-10 flex px-4 sm:px-8 pt-16"
        style={{ zIndex: 1, height: 'calc(100vh - 200px)' }}
      >
        {/* Left Column - Video Player (top) and Upload Area (bottom) - 1/3 width */}
        <div className="w-1/3 pr-4 flex flex-col">
          {/* Video Player - 16:9 ratio at top */}
          {uploadedFile && (
            <div className="mb-4">
              <h2 className="text-lg font-bold text-gray-800 text-center mb-4">Video Player</h2>
              <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-2">
                <div className="relative w-full" style={{ aspectRatio: '16/9' }}>
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    className="w-full h-full object-cover rounded-lg"
                    onTimeUpdate={handleVideoTimeUpdate}
                    onLoadedMetadata={handleVideoLoadedMetadata}
                    onPlay={async () => {
                      setIsPlaying(true)

                      // Resume audio context if suspended
                      if (audioContext && audioContext.state === 'suspended') {
                        await audioContext.resume()
                      }

                      // Ensure audio context is properly connected
                      if (videoRef.current && audioContext) {
                        try {
                          // If we don't have an audio source or it's disconnected, recreate it
                          if (!audioSource || !audioSource.context) {
                            const source = audioContext.createMediaElementSource(videoRef.current)
                            setAudioSource(source)

                            // Create analyser if not exists
                            if (!analyser) {
                              const analyserNode = audioContext.createAnalyser()
                              analyserNode.fftSize = 2048
                              analyserNode.smoothingTimeConstant = 0.8
                              setAnalyser(analyserNode)
                            }

                            // Connect audio graph
                            source.connect(analyser)
                            analyser.connect(audioContext.destination)
                          }
                        } catch (error) {
                          console.error('Error setting up audio on play:', error)
                        }
                      }
                    }}
                    onPause={() => setIsPlaying(false)}
                    controls
                    preload="metadata"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Upload Area - Takes remaining space */}
          <div className="flex-1 min-h-0">
            <h2 className="text-lg font-bold text-gray-800 text-center mb-4">Upload Video</h2>
            <div
              className={`border-2 border-dashed rounded-xl p-4 text-center transition-all duration-300 h-full flex flex-col justify-center ${
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
                <div className="space-y-3">
                  <div className="w-12 h-12 bg-green-500 rounded-full flex items-center justify-center mx-auto">
                    <svg
                      className="w-6 h-6 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-gray-800 mb-1">File Uploaded!</h3>
                    <p className="text-gray-600 mb-1 text-sm truncate">{uploadedFile.name}</p>
                    <p className="text-xs text-gray-500">
                      {(uploadedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      setUploadedFile(null)
                    }}
                    className="text-xs text-gray-500 hover:text-gray-700 underline"
                  >
                    Remove file
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="w-12 h-12 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center mx-auto">
                    <svg
                      className="w-6 h-6 text-white"
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
                  <div>
                    <h3 className="text-lg font-bold text-gray-800 mb-1">
                      {isDragOver ? 'Drop here' : 'Upload Video'}
                    </h3>
                    <p className="text-gray-600 mb-2 text-sm">Click or drag file</p>
                    <p className="text-xs text-gray-500">MP4, MOV, AVI</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column - Audio Visualizer - 2/3 width */}
        {uploadedFile && (
          <div className="w-2/3 pl-4">
            <h2 className="text-2xl font-bold text-gray-800 text-center mb-4">Audio Analysis</h2>
            <div className="bg-black rounded-lg shadow-lg border border-gray-200 overflow-hidden h-full">
              <FluidVisualizer
                analyser={analyser}
                audioContext={audioContext}
                isPlaying={isPlaying}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default UploadVideoPage
