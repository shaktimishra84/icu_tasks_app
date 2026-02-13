# ICU Task Assistant (MVP)

This project is a local MVP for ICU workflow support.

It lets you:
- Upload clinical resources (guidelines/books in PDF, DOCX, TXT, image formats)
- Upload or paste patient case data
- Generate a de-identified case summary and suggested next tasks
- Flag potentially missed tests, imaging, or consultations

## Important Safety Note

This tool is for workflow support only.  
It is **not** a diagnosis engine or autonomous medical decision system.  
A licensed clinician must review all output before any action.

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional but recommended) set OpenAI key:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4.1-mini"
```

4. (Optional) restrict access to specific users:

```bash
export ALLOWED_USERS="user1@example.com,user2@example.com,user3@example.com,user4@example.com"
```

5. Run the app:

```bash
streamlit run app.py
```

## How It Works

- Resource files are parsed and stored in `data/resources.json`.
- A lightweight local retrieval step finds the most relevant resources for a case.
- The analysis engine:
  - De-identifies summary content
  - Suggests next tasks (short horizon + 24h horizon)
  - Highlights possible missed workup items
- If no `OPENAI_API_KEY` is set, the app falls back to a generic rules-based checklist.
- All-beds output can be exported as a print-ready rounds PDF (`output/ICU_Rounds_<date>_<shift>.pdf`) using ReportLab.
- If `ALLOWED_USERS` is set, app access is restricted to signed-in users in that allow-list.
- On fresh cloud deploys, click **Rebuild startup index** once to build the PDF resource index.

## GitHub Deployment (4 users)

1. Push this project to a private GitHub repo.
2. Deploy it on Streamlit Community Cloud from that repo.
3. In Streamlit app settings, add secrets:
   - Use the exact template in `.streamlit/secrets.toml.example`.
   - `ALLOWED_USERS` currently prefilled for:
     - `samir.jj.ax@gmail.com`
     - `dash.abhilash2012@gmail.com`
     - `drsatyajit87@gmail.com`
     - `shaktimishra84@gmail.com`
   - Fill `[auth]` values from your Google OIDC app.
4. Share the Streamlit app URL with those 4 users only.

### Streamlit Cloud Click Path

1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in.
2. Click **New app**.
3. Select repo: `shaktimishra84/icu_tasks_app`, branch: `main`, main file: `app.py`.
4. Open **Advanced settings** and paste secrets from `.streamlit/secrets.toml.example` (with real `[auth]` values).
5. Click **Deploy**.
6. Open the deployed URL and verify:
   - non-listed email -> denied
   - listed email -> app loads

Notes:
- Do not commit `.env` (already in `.gitignore`).
- If `ALLOWED_USERS` is empty, the app stays open-access.

## Supported File Types

- Resources: `.pdf`, `.docx`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`
- Patient case: same as above

Notes:
- Legacy `.doc` files are not supported directly (convert to `.docx` first).
- Image OCR uses `pytesseract`; system Tesseract installation may be required.

## Next Improvements

- Add stronger PHI de-identification using structured redaction checks
- Add section-aware parsing for long textbooks/guidelines
- Add audit log + feedback loop for clinician corrections
- Add authentication and encrypted storage for production use
