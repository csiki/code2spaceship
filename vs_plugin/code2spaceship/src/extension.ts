import * as vscode from 'vscode';
import * as opencv from 'opencv';



// CHAT GPT CODE


// This function uses OpenCV to detect the outlines of code in the given image
// and applies the spaceship filter to them. It returns the transformed image.
	async function applySpaceshipFilter(image: opencv.Mat): Promise<opencv.Mat> {
	// Use OpenCV's edge detection algorithms to identify the outlines of the code
	const edges = image.canny(0, 100);

	// Load the spaceship image and resize it to match the size of the code image
	const spaceship = await opencv.imreadAsync('spaceship.png');
	spaceship.resize(image.cols, image.rows);

	// Overlay the spaceship image on top of the code outlines and blend the two images together
	const filtered = edges.addWeighted(spaceship, 0.5, 0.5);
	return filtered;
}

// This function creates a new VS Code editor window that displays the code text
// and the spaceship image behind it.
async function showCodeWithSpaceship(code: string, image: opencv.Mat): Promise<void> {
  // Create a new editor window using the Extension API
  const panel = vscode.window.createWebviewPanel(
    'codeWithSpaceship',
    'Code with Spaceship',
    vscode.ViewColumn.One,
    {
      enableScripts: true,
      retainContextWhenHidden: true,
    }
  );

  // Set the HTML content of the editor window to display the code text
  // and the spaceship image behind it
  panel.webview.html = `
    <!DOCTYPE html>
    <html>
      <head>
        <style>
          body {
            font-family: monospace;
            background-image: url('${image.toBase64()}');
            background-repeat: no-repeat;
            background-position: center;
          }
        </style>
      </head>
      <body>
        <pre>${code}</pre>
      </body>
    </html>
  `;
}

export function activate(context: vscode.ExtensionContext) {
	// Register a command that can be triggered by the user to apply the spaceship
	// filter to an image of code and display the resulting image in a new editor window
	context.subscriptions.push(vscode.commands.registerCommand(
	  'extension.showCodeWithSpaceship',
	  async () => {
		// Get the currently selected image file in VS Code
		const imageFile = vscode.window.activeTextEditor?.document;
		if (!imageFile || imageFile.languageId !== 'image') {
		  vscode.window.showErrorMessage('No image file is selected.');
		  return;
		}
  
		// Read the selected image file and apply the spaceship filter to it
		const image = await opencv.imreadAsync(imageFile.uri.fsPath);
		const filteredImage = await applySpaceshipFilter(image);
  
		// Open the image file in a new editor window and display the filtered image behind the code
		await vscode.workspace.openTextDocument(imageFile.uri);
		const code = vscode.window.activeTextEditor?.document.getText();
		if (code) {
		  await showCodeWithSpaceship(code, filteredImage);
		}
	  }
	));
}


  export function deactivate() {}









// OWN CODE


// // The module 'vscode' contains the VS Code extensibility API
// // Import the module and reference it with the alias vscode in your code below
// import * as vscode from 'vscode';

// // This method is called when your extension is activated
// // Your extension is activated the very first time the command is executed
// export function activate(context: vscode.ExtensionContext) {

// 	// Use the console to output diagnostic information (console.log) and errors (console.error)
// 	// This line of code will only be executed once when your extension is activated
// 	console.log('Congratulations, your extension "code2spaceship" is now active!');

// 	// The command has been defined in the package.json file
// 	// Now provide the implementation of the command with registerCommand
// 	// The commandId parameter must match the command field in package.json
// 	let disposable = vscode.commands.registerCommand('code2spaceship.helloWorld', () => {

// 		vscode.window.showInformationMessage('Spaceshipifying code in 5..');
// 		// TODO count down hard coded from 5 and if not loaded say "just kidding, not yet..", if loaded before "nevermind done"
// 		//   run another function here as a coroutine in 1 sec that checks the state of bg loaded, and is scheduled recursively every second
		
// 		let command = "#{@pypath} #{@shipifyPath}"
// 		command += " --steps #{@diffusionSteps}"
// 		command += " --size #{@spaceshipSize}"
// 		command += " --prompt \"#{@spaceshipStyle}\""
// 		command += " \"#{codePath}\" \"#{@spaceshipImgPath}\""
// 		activeEditor.insertText(command + '\n')

// 	});

// 	context.subscriptions.push(disposable);
// }

// // This method is called when your extension is deactivated
// export function deactivate() {}
