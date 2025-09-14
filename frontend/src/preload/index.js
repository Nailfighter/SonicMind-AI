import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'

// Custom APIs for renderer
const api = {
  getBackendData: (route = 'time') => ipcRenderer.invoke('get-backend-data', route),
  getTimeData: () => ipcRenderer.invoke('get-backend-data', 'time'),
  getRandomNumber: () => ipcRenderer.invoke('get-backend-data', 'random'),
  getSystemInfo: () => ipcRenderer.invoke('get-backend-data', 'system'),
  getWeatherData: () => ipcRenderer.invoke('get-backend-data', 'weather'),
  processAudio: (audioData, filename, processType) =>
    ipcRenderer.invoke('process-audio', audioData, filename, processType)
}

// Use `contextBridge` APIs to expose Electron APIs to
// renderer only if context isolation is enabled, otherwise
// just add to the DOM global.
if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('api', api)
  } catch (error) {
    console.error(error)
  }
} else {
  window.electron = electronAPI
  window.api = api
}
