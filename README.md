# Document Line-Break Fixer

This program detects and fixes unintended line breaks in Word documents (primarily `.docx`) that often arise from PDF-to-DOCX conversions (and soon EPUB conversions). It streamlines your document’s text by merging paragraphs that should be one continuous sentence.

## Screenshot

![Screenshot](https://github.com/ShenSheiBot/docbreak_fixer/raw/master/screenshot.png)

## What This Program Does

- **Detect** paragraphs that accidentally got split into two lines.
- **Preview** where these issues occur.
- **Fix** those breaks automatically by merging the paragraphs into a single paragraph.

**Planned Feature**: Support for `.epub` files in the future.

---

## Installation & Setup

### 1. Install [Poetry](https://python-poetry.org/docs/#installation)

If you do not already have [Poetry](https://python-poetry.org/) installed, please follow their [official instructions](https://python-poetry.org/docs/#installation) to install it on your system.

**Example (for macOS/Linux)**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Then add the Poetry command to your PATH (the installer will provide instructions).

### 2. Clone or Download This Repository

```bash
git clone https://github.com/yourusername/line-break-fixer.git
cd line-break-fixer
```
*(Or simply download the code as a ZIP and unzip into a folder.)*

### 3. Install Dependencies with Poetry

Inside the project folder, run:
```bash
poetry install
```
This will create a virtual environment and install all necessary libraries.

### 4. Launch the Program

After installing the dependencies, you have two options:

- **Option A**: Run with Poetry directly:
  ```bash
  poetry run python main.py
  ```
  
- **Option B**: Enter the Poetry shell and run `python main.py`:
  ```bash
  poetry shell
  python main.py
  ```

You should see console output indicating the server is running on `http://0.0.0.0:28000`.

### 5. Open Your Browser

Visit [http://localhost:28000](http://localhost:28000) in your browser.  
You’ll see the program’s interface:
1. Choose a `.docx` file (preferably one converted from PDF).
2. Click “Detect Issues” to preview and select all issues.
3. Click “Download Fixed Document” to retrieve the corrected file.

---

## Using an LLM (Optional)

By default, this app has a **Rule-based Detector** that looks for potential line-break errors in your documents. If you’d like to use a Large Language Model (LLM) for more nuanced detection, you can enable the **LLM Detector** in the app’s Settings panel.

**Supported LLMs and how to configure them**:
- If you check **“LLM Detector”**, you must specify:
  - **Model Name**  
  - **Environment Variables** (for example, your API key)

To see a full list of supported models and environment variables, check [the LiteLLM OpenAI docs](https://docs.litellm.ai/docs/providers/openai).  

### Example: OpenAI GPT-3.5

In the Settings panel:
- **Model Name**: `openai/gpt-3.5-turbo`
- **Environment Variables**:
  - Name: `OPENAI_API_KEY`
  - Value: `sk-your-openai-key`

### Example: Gemini (hypothetical)

If Gemini (Google’s upcoming model) becomes available via LiteLLM, you might do:
- **Model Name**: `google/gemini`
- **Environment Variables**:
  - Name: `GOOGLE_API_KEY`
  - Value: `your-google-api-key`

*(These details may change based on actual Gemini availability—refer to the official provider docs when it launches.)*

---

## Tips

1. **Selecting/Unselecting Issues**  
   - You can individually click “Select” next to paragraphs in the preview to merge them with the following paragraph.  
   - Or use “Select All Issues” if you trust the detection.  
   - **Yellow** highlighting indicates an automatically detected issue; **Red** indicates you’ve selected it to merge.

2. **Logs**  
   - On the left side, you’ll see a “Logs” panel that shows real-time messages. If you enable LLM detection, it will display messages about the AI detection process.

3. **Common Errors**  
   - If you see an error about missing API keys, ensure you have set your environment variables in the UI or your system shell.

---

## Contact & Contributions

- If you encounter bugs or want to contribute, please open an issue or pull request on GitHub.
- Feedback is welcome, especially regarding EPUB support or new rule-based scenarios for line-break detection.

Thank you for using Document Line-Break Fixer! We hope it saves you time cleaning up messy PDF conversions.
