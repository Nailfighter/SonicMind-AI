import { app, shell, BrowserWindow, ipcMain } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { spawn } from 'child_process'

function createWindow() {
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false,
      webSecurity: true,
      contextIsolation: true,
      nodeIntegration: false,
      allowRunningInsecureContent: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // IPC test
  ipcMain.on('ping', () => {})

  // IPC handler for calling Python backend with route support
  ipcMain.handle('get-backend-data', async (event, route = 'time') => {
    return new Promise((resolve, reject) => {
      const pythonScript = join(__dirname, '../../../backend/routes.py')
      const pythonProcess = spawn('python', [pythonScript, route])

      let data = ''
      let error = ''

      pythonProcess.stdout.on('data', (chunk) => {
        data += chunk.toString()
      })

      pythonProcess.stderr.on('data', (chunk) => {
        error += chunk.toString()
      })

      pythonProcess.on('close', (code) => {
        if (code === 0) {
          try {
            const result = JSON.parse(data)
            resolve(result)
          } catch (parseError) {
            reject(new Error(`Failed to parse Python output: ${parseError.message}`))
          }
        } else {
          reject(new Error(`Python script failed with code ${code}: ${error}`))
        }
      })

      pythonProcess.on('error', (err) => {
        reject(new Error(`Failed to start Python process: ${err.message}`))
      })
    })
  })

  // IPC handler for audio processing
  ipcMain.handle('process-audio', async (event, audioData, filename, processType) => {
    return new Promise((resolve, reject) => {
      const fs = require('fs')
      const path = require('path')
      const os = require('os')

      try {
        // Create a temporary file for the audio data
        const tempDir = os.tmpdir()
        const tempFileName = `audio_${Date.now()}_${Math.random().toString(36).substr(2, 9)}.tmp`
        const tempFilePath = path.join(tempDir, tempFileName)

        // Decode base64 data and write to temporary file
        const audioBuffer = Buffer.from(audioData, 'base64')
        fs.writeFileSync(tempFilePath, audioBuffer)

        const pythonScript = join(__dirname, '../../../backend/routes.py')
        const pythonProcess = spawn('python', [
          pythonScript,
          'audio',
          tempFilePath,
          filename,
          processType
        ])

        let data = ''
        let error = ''

        pythonProcess.stdout.on('data', (chunk) => {
          data += chunk.toString()
        })

        pythonProcess.stderr.on('data', (chunk) => {
          error += chunk.toString()
        })

        pythonProcess.on('close', (code) => {
          // Clean up temporary file
          try {
            if (fs.existsSync(tempFilePath)) {
              fs.unlinkSync(tempFilePath)
            }
          } catch (cleanupError) {
            // Silently handle cleanup errors
          }

          if (code === 0) {
            try {
              const result = JSON.parse(data)
              resolve(result)
            } catch (parseError) {
              reject(new Error(`Failed to parse Python output: ${parseError.message}`))
            }
          } else {
            reject(new Error(`Python script failed with code ${code}: ${error}`))
          }
        })

        pythonProcess.on('error', (err) => {
          // Clean up temporary file on error
          try {
            if (fs.existsSync(tempFilePath)) {
              fs.unlinkSync(tempFilePath)
            }
          } catch (cleanupError) {
            // Silently handle cleanup errors
          }
          reject(new Error(`Failed to start Python process: ${err.message}`))
        })
      } catch (writeError) {
        reject(new Error(`Failed to write temporary file: ${writeError.message}`))
      }
    })
  })

  createWindow()

  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and require them here.
