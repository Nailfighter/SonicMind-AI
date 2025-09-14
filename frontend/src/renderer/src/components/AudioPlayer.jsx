import React, { useRef, useState, useCallback } from 'react'
import * as Tone from 'tone'

const AudioPlayer = ({ onAudioLoad }) => {
  // Channel 1 state
  const [isPlaying1, setIsPlaying1] = useState(false)
  const [isLoading1, setIsLoading1] = useState(false)
  const [fileName1, setFileName1] = useState(null)
  const [duration1, setDuration1] = useState(0)
  const [currentTime1, setCurrentTime1] = useState(0)
  const fileInputRef1 = useRef(null)
  const playerRef1 = useRef(null)
  const analyserRef1 = useRef(null)
  const intervalRef1 = useRef(null)

  // Channel 2 state
  const [isPlaying2, setIsPlaying2] = useState(false)
  const [isLoading2, setIsLoading2] = useState(false)
  const [fileName2, setFileName2] = useState(null)
  const [duration2, setDuration2] = useState(0)
  const [currentTime2, setCurrentTime2] = useState(0)
  const fileInputRef2 = useRef(null)
  const playerRef2 = useRef(null)
  const analyserRef2 = useRef(null)
  const intervalRef2 = useRef(null)

  const handleFileSelect = useCallback(
    async (event, channel) => {
      const file = event.target.files?.[0]
      if (!file) return

      const setIsLoading = channel === 1 ? setIsLoading1 : setIsLoading2
      const setFileName = channel === 1 ? setFileName1 : setFileName2
      const setDuration = channel === 1 ? setDuration1 : setDuration2
      const playerRef = channel === 1 ? playerRef1 : playerRef2
      const analyserRef = channel === 1 ? analyserRef1 : analyserRef2

      setIsLoading(true)
      setFileName(file.name)

      try {
        // Ensure audio context is started
        await Tone.start()

        // Dispose of previous player and analyser
        if (playerRef.current) {
          playerRef.current.dispose()
        }
        if (analyserRef.current) {
          analyserRef.current.dispose()
        }

        // Create URL for the audio file
        const audioUrl = URL.createObjectURL(file)

        // Create new player and analyser
        const player = new Tone.Player({
          url: audioUrl,
          onload: () => {
            setDuration(player.buffer.duration)
            setIsLoading(false)
          },
          onerror: (error) => {
            console.error(`Error loading audio for channel ${channel}:`, error)
            setIsLoading(false)
          }
        })

        const analyser = new Tone.Analyser('fft', 256)

        // Connect player to analyser and to destination
        player.connect(analyser)
        player.toDestination()

        playerRef.current = player
        analyserRef.current = analyser

        // Notify parent component
        onAudioLoad(channel, player, analyser)
      } catch (error) {
        console.error(`Error setting up audio for channel ${channel}:`, error)
        setIsLoading(false)
      }
    },
    [onAudioLoad]
  )

  const togglePlayback = useCallback(
    async (channel) => {
      const playerRef = channel === 1 ? playerRef1 : playerRef2
      const isPlaying = channel === 1 ? isPlaying1 : isPlaying2
      const setIsPlaying = channel === 1 ? setIsPlaying1 : setIsPlaying2
      const setCurrentTime = channel === 1 ? setCurrentTime1 : setCurrentTime2
      const intervalRef = channel === 1 ? intervalRef1 : intervalRef2

      if (!playerRef.current) return

      try {
        await Tone.start()

        if (isPlaying) {
          playerRef.current.stop()
          setIsPlaying(false)
          if (intervalRef.current) {
            clearInterval(intervalRef.current)
            intervalRef.current = null
          }
        } else {
          playerRef.current.start()
          setIsPlaying(true)

          // Update current time
          intervalRef.current = setInterval(() => {
            if (playerRef.current && playerRef.current.state === 'started') {
              setCurrentTime(Tone.Transport.seconds)
            }
          }, 100)
        }
      } catch (error) {
        console.error(`Error toggling playback for channel ${channel}:`, error)
      }
    },
    [isPlaying1, isPlaying2]
  )

  const handleSeek = useCallback((event, channel) => {
    const playerRef = channel === 1 ? playerRef1 : playerRef2
    const setCurrentTime = channel === 1 ? setCurrentTime1 : setCurrentTime2

    if (!playerRef.current) return

    const seekTime = parseFloat(event.target.value)
    playerRef.current.seek(seekTime)
    setCurrentTime(seekTime)
  }, [])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const renderChannelControls = (channel) => {
    const isLoading = channel === 1 ? isLoading1 : isLoading2
    const fileName = channel === 1 ? fileName1 : fileName2
    const isPlaying = channel === 1 ? isPlaying1 : isPlaying2
    const currentTime = channel === 1 ? currentTime1 : currentTime2
    const duration = channel === 1 ? duration1 : duration2
    const fileInputRef = channel === 1 ? fileInputRef1 : fileInputRef2
    const playerRef = channel === 1 ? playerRef1 : playerRef2
    const channelColor = channel === 1 ? '#4CAF50' : '#FF9800'

    return (
      <div
        style={{
          background: 'rgba(0, 0, 0, 0.7)',
          padding: '15px',
          borderRadius: '8px',
          border: `2px solid ${channelColor}`,
          marginBottom: '10px'
        }}
      >
        <h3 style={{ margin: '0 0 10px 0', color: channelColor, fontSize: '16px' }}>
          Channel {channel}
        </h3>

        <div style={{ marginBottom: '10px' }}>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={(e) => handleFileSelect(e, channel)}
            style={{ display: 'none' }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            style={{
              background: channelColor,
              color: 'white',
              border: 'none',
              padding: '8px 16px',
              borderRadius: '5px',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              fontSize: '12px'
            }}
          >
            {isLoading ? 'Loading...' : 'Load Audio'}
          </button>
        </div>

        {fileName && (
          <div>
            <div style={{ fontSize: '12px', marginBottom: '8px', color: '#ccc' }}>
              <strong>File:</strong> {fileName}
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
              <button
                onClick={() => togglePlayback(channel)}
                disabled={!playerRef.current || isLoading}
                style={{
                  background: isPlaying ? '#f44336' : '#2196F3',
                  color: 'white',
                  border: 'none',
                  padding: '6px 12px',
                  borderRadius: '4px',
                  cursor: !playerRef.current || isLoading ? 'not-allowed' : 'pointer',
                  fontSize: '12px'
                }}
              >
                {isPlaying ? '⏸️' : '▶️'}
              </button>

              <span style={{ fontSize: '11px', color: '#ccc' }}>
                {formatTime(currentTime)} / {formatTime(duration)}
              </span>
            </div>

            {duration > 0 && (
              <input
                type="range"
                min="0"
                max={duration}
                value={currentTime}
                onChange={(e) => handleSeek(e, channel)}
                style={{
                  width: '100%',
                  height: '4px',
                  background: '#555',
                  outline: 'none',
                  borderRadius: '2px'
                }}
              />
            )}
          </div>
        )}
      </div>
    )
  }

  return (
    <div
      className="audio-player"
      style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        width: '300px',
        background: 'rgba(0, 0, 0, 0.9)',
        padding: '15px',
        borderRadius: '10px',
        color: 'white',
        fontFamily: 'Arial, sans-serif',
        zIndex: 1000
      }}
    >
      <h2 style={{ margin: '0 0 15px 0', fontSize: '18px', textAlign: 'center' }}>
        Dual Audio Player
      </h2>

      {renderChannelControls(1)}
      {renderChannelControls(2)}
    </div>
  )
}

export default AudioPlayer
