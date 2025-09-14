import React, { useState, useRef, useEffect, useCallback } from 'react'

// The main Vertical EQ Slider Component
const VerticalEQSlider = ({
  initialValue = 50,
  min = 0,
  max = 100,
  onChange = () => {},
  label = 'LEVEL',
  disabled = false
}) => {
  // State for the slider's current value
  const [value, setValue] = useState(initialValue)
  // State to track when the user is actively dragging the slider
  const [isDragging, setIsDragging] = useState(false)

  // Update value when initialValue changes (from backend)
  useEffect(() => {
    if (!isDragging) {
      setValue(initialValue)
    }
  }, [initialValue, isDragging])

  // Ref to the slider's main container element
  const sliderRef = useRef(null)
  // Ref to store the callback function to avoid re-adding event listeners
  const onChangeRef = useRef(onChange)

  // Update the ref if the onChange prop changes
  useEffect(() => {
    onChangeRef.current = onChange
  }, [onChange])

  // Calculate the percentage position of the thumb based on the current value
  const getPercentage = useCallback(() => {
    // Clamp the value to ensure it's within the min/max bounds
    const clampedValue = Math.max(min, Math.min(value, max))
    return ((clampedValue - min) / (max - min)) * 100
  }, [value, min, max])

  // Function to handle value changes from mouse or touch events
  const handleValueChange = useCallback(
    (event) => {
      if (!sliderRef.current) return

      // Get the bounding box of the slider element
      const rect = sliderRef.current.getBoundingClientRect()
      // Determine the vertical position (clientY) from either mouse or touch event
      const clientY = 'touches' in event ? event.touches[0].clientY : event.clientY

      // Calculate the relative vertical position within the slider
      const relativeY = clientY - rect.top

      // Calculate the new value based on the position, inverting it for vertical slider
      let newValue = ((rect.height - relativeY) / rect.height) * (max - min) + min

      // Clamp the new value to be within the min/max bounds
      newValue = Math.max(min, Math.min(newValue, max))

      setValue(newValue)
      onChangeRef.current(newValue)
    },
    [min, max]
  )

  // --- Event Handlers ---

  // Handle mouse down and touch start events to begin dragging
  const handleInteractionStart = useCallback(
    (event) => {
      if (disabled) return
      
      document.body.style.cursor = 'grabbing'
      setIsDragging(true)
      handleValueChange(event)
    },
    [handleValueChange, disabled]
  )

  // Handle mouse move and touch move events during dragging
  const handleInteractionMove = useCallback(
    (event) => {
      if (isDragging) {
        handleValueChange(event)
      }
    },
    [isDragging, handleValueChange]
  )

  // Handle mouse up and touch end events to stop dragging
  const handleInteractionEnd = useCallback(() => {
    document.body.style.cursor = 'default'
    setIsDragging(false)
  }, [])

  // --- Effect for Global Event Listeners ---

  useEffect(() => {
    // Add event listeners to the window to handle dragging even if cursor leaves the slider
    window.addEventListener('mousemove', handleInteractionMove)
    window.addEventListener('mouseup', handleInteractionEnd)
    window.addEventListener('touchmove', handleInteractionMove, { passive: false })
    window.addEventListener('touchend', handleInteractionEnd)

    // Cleanup function to remove event listeners when the component unmounts
    return () => {
      window.removeEventListener('mousemove', handleInteractionMove)
      window.removeEventListener('mouseup', handleInteractionEnd)
      window.removeEventListener('touchmove', handleInteractionMove)
      window.removeEventListener('touchend', handleInteractionEnd)
      document.body.style.cursor = 'default'
    }
  }, [handleInteractionMove, handleInteractionEnd])

  const percentage = getPercentage()

  return (
    <div className={`flex flex-col items-center justify-center pt-2 pb-2 px-2 select-none space-y-2 rounded-2xl shadow-lg transition-all duration-300 ease-out border ${
      disabled 
        ? 'bg-gray-200/30 border-gray-300/50 opacity-50' 
        : 'bg-gray-100/50 backdrop-blur-xl border-white/50'
    }`}>
      <div
        className={`relative w-6 h-48 flex justify-center ${
          disabled ? 'cursor-not-allowed' : 'cursor-grab'
        }`}
        onMouseDown={handleInteractionStart}
        onTouchStart={handleInteractionStart}
        ref={sliderRef}
      >
        {/* Zero Marker Line */}
        <div
          className="absolute left-1/2 -translate-x-1/2 w-16 border-t-2 border-dashed border-gray-500 opacity-75"
          style={{ bottom: `${((0 - min) / (max - min)) * 100}%` }}
        />

        {/* Slider Track */}
        <div className={`relative w-1.5 h-full rounded-full overflow-hidden shadow-inner ${
          disabled ? 'bg-gray-400' : 'bg-gray-300'
        }`}>
          {/* Filled part of the track showing the current value */}
          <div
            className={`absolute bottom-0 w-full rounded-full transition-all duration-75 ease-out ${
              disabled ? 'bg-gray-600' : 'bg-black'
            }`}
            style={{ height: `${percentage}%` }}
          />
        </div>

        {/* Slider Thumb */}
        <div
          className={`absolute w-8 h-6 rounded border-t-2 transition-all duration-75 ease-out transform -translate-x-1/2 ${
            disabled 
              ? 'bg-gradient-to-b from-gray-600 to-gray-700 border-gray-400/20 shadow-lg shadow-gray-500/30'
              : `bg-gradient-to-b from-gray-800 to-black border-white/20 ${
                  isDragging ? 'scale-125 shadow-2xl shadow-black/40' : 'shadow-lg shadow-black/30'
                }`
          }`}
          style={{
            bottom: `calc(${percentage}% - 12px)`, // Center thumb (24px height / 2)
            left: '50%'
          }}
        />
      </div>

      {/* Label and Value Display */}
      <div className="flex items-center justify-between w-24 pt-1">
        <div className={`text-sm font-semibold uppercase tracking-wider ${
          disabled ? 'text-gray-400' : 'text-gray-500'
        }`}>{label}</div>
        <div key={Math.round(value)} className={`text-lg font-semibold animate-pop-in ${
          disabled ? 'text-gray-600' : 'text-black'
        }`}>
          {value.toFixed(1)}
        </div>
      </div>
    </div>
  )
}

// Export the VerticalEQSlider as the default component
export default VerticalEQSlider
