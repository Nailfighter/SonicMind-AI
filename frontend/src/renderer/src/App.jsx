import React, { useState, useCallback } from 'react'
import * as Tone from 'tone'
import AudioPlayer from './components/AudioPlayer'
import FluidVisualizer from './components/FluidVisualizer'

const App = () => {
  const [analyser1, setAnalyser1] = useState(undefined)
  const [analyser2, setAnalyser2] = useState(undefined)
  const [audioSource, setAudioSource] = useState('file')

  const handleAudioLoad = useCallback((channel, player, newAnalyser) => {
    if (channel === 1) {
      setAnalyser1(newAnalyser)
    } else {
      setAnalyser2(newAnalyser)
    }
    setAudioSource('file')
  }, [])

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        position: 'relative',
        background: 'black',
        overflow: 'hidden'
      }}
    >
      <FluidVisualizer analyser1={analyser1} analyser2={analyser2} audioSource={audioSource} />

      <AudioPlayer onAudioLoad={handleAudioLoad} />
    </div>
  )
}

export default App
