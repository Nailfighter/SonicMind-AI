import React from 'react'
import FluidVisualizer from './components/FluidVisualizer'

const App = () => {
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
      <FluidVisualizer />
    </div>
  )
}

export default App
