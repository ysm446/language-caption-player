const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  // ファイルダイアログ
  openVideo: ()         => ipcRenderer.invoke('dialog:openVideo'),
  openSrt:   ()         => ipcRenderer.invoke('dialog:openSrt'),

  // ファイル読み込み
  readFile:  (filePath) => ipcRenderer.invoke('fs:readFile', filePath),

  // プレイヤーウィンドウを開く
  openPlayer: (args)    => ipcRenderer.invoke('window:openPlayer', args),
})
