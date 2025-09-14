import { contextBridge } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Note: Socket.IO-based API is now handled directly in the renderer process
// This is a transitional approach that maintains the same API surface
// while the renderer uses BackendAPI service for Socket.IO communication

const api = {
  // Legacy IPC methods - now return promises that will be handled by BackendAPI
  // Components should migrate to using BackendAPI directly for better real-time features
  
  getBackendData: (route = 'time') => {
    // This will be handled by BackendAPI in the renderer
    console.warn('api.getBackendData is deprecated. Use BackendAPI.getBackendData() directly in components')
    return Promise.reject(new Error('Use BackendAPI service instead of preload IPC'))
  },
  
  getTimeData: () => {
    console.warn('api.getTimeData is deprecated. Use BackendAPI.getTimeData() directly in components')
    return Promise.reject(new Error('Use BackendAPI service instead of preload IPC'))
  },
  
  getRandomNumber: () => {
    console.warn('api.getRandomNumber is deprecated. Use BackendAPI.getRandomNumber() directly in components')
    return Promise.reject(new Error('Use BackendAPI service instead of preload IPC'))
  },
  
  getSystemInfo: () => {
    console.warn('api.getSystemInfo is deprecated. Use BackendAPI.getSystemInfo() directly in components')
    return Promise.reject(new Error('Use BackendAPI service instead of preload IPC'))
  },
  
  getWeatherData: () => {
    console.warn('api.getWeatherData is deprecated. Use BackendAPI.getWeatherData() directly in components')
    return Promise.reject(new Error('Use BackendAPI service instead of preload IPC'))
  },
  
  processAudio: (audioData, filename, processType) => {
    console.warn('api.processAudio is deprecated. Use BackendAPI.processAudio() directly in components')
    return Promise.reject(new Error('Use BackendAPI service instead of preload IPC'))
  }
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    // Silently handle API errors
  }
} else {
  window.electron = electronAPI
  window.api = api
}
