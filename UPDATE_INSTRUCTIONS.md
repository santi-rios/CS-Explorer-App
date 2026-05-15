# How to Update the Chemical Space Explorer App

## Overview
This document explains how to update the Chemical Space Explorer Shiny application with new versions of the article and website links. Follow these steps carefully, even if you're not very technical—we've made it as simple as possible!

---

## What Changed?
The app has been updated with:
1. **New article DOI**: Changed from ChemRxiv preprint to published version
   - Old: `https://chemrxiv.org/engage/chemrxiv/article-details/67920ada6dde43c908f688f6`
   - New: `https://doi.org/10.1162/QSS.a.471`

2. **New PDF file**: Updated article PDF location
   - Old: `www/original_article.pdf`
   - New: `www/qss.a.471.pdf`

3. **New landing page**: Updated the main index.html with buttons for easy navigation
   - App link: `https://countrywise-chemical-space.com/CS-Explorer-App/`
   - Article link: `https://doi.org/10.1162/QSS.a.471`

---

## Step-by-Step Instructions

### Prerequisites
Before you start, make sure you have:
- Access to the Azure server running the Shiny app
- Basic command-line familiarity (or willingness to follow instructions exactly)
- Git installed on the server
- Python 3.8+ installed

### Option A: Using Git (Recommended - Easiest)

**Step 1: Connect to the Azure Server**
```bash
ssh your-username@your-server-ip
```

**Step 2: Navigate to the App Directory**
```bash
cd /home/santi/projects/ShinyServer_Azure/ShinyServer_Azure/shiny-apps-backup/CS-Explorer-App
```

**Step 3: Download the Latest Changes**
```bash
git pull origin main
```

This command will automatically update all files with the new article links, new PDF reference, and new landing page.

**Step 4: Restart the Shiny Server**
```bash
sudo systemctl restart shiny-server
```

Done! The app should now show the new article link and landing page.

---

### Option B: Manual Update (If Git is Unavailable)

If you don't have Git access or prefer to update manually:

**Step 1: Back up current files**
```bash
cd /home/santi/projects/ShinyServer_Azure/ShinyServer_Azure/shiny-apps-backup/CS-Explorer-App
cp app.py app.py.backup
cp ../index.html ../index.html.backup
```

**Step 2: Edit app.py file**
1. Open `app.py` in a text editor
2. Find these three occurrences of the old ChemRxiv URL:
   ```
   https://chemrxiv.org/engage/chemrxiv/article-details/67920ada6dde43c908f688f6
   ```
3. Replace each with:
   ```
   https://doi.org/10.1162/QSS.a.471
   ```

*Locations in app.py:*
- Line ~108: In the article citation
- Line ~329-330: In the "Useful Links" footer section

**Step 3: Replace the PDF file reference** (if needed)
1. Find the commented line in app.py that mentions `original_article.pdf`
2. Update it to reference `qss.a.471.pdf` (this is optional since this code is currently commented out)

**Step 4: Update index.html**
1. Open the `index.html` file in `/srv/shiny-server/` or your deployment directory
2. Update the button links:
   - Change the article URL in the "Original Article" button to: `https://doi.org/10.1162/QSS.a.471`

**Step 5: Ensure the new PDF is in place**
1. Verify that `www/qss.a.471.pdf` exists in the app directory
2. If not, copy/move the new PDF file there

**Step 6: Restart the Shiny Server**
```bash
sudo systemctl restart shiny-server
```

---

## Verification Checklist

After completing the update, verify everything works:

**✓ Check 1: Landing Page**
- [ ] Visit your main portal page (usually at your server URL)
- [ ] Confirm the two buttons are visible: "Chemical Space Explorer Interactive App" and "Original Article"
- [ ] Click the "Original Article" button and confirm it opens the DOI link: https://doi.org/10.1162/QSS.a.471

**✓ Check 2: In-App Links**
- [ ] Open the Chemical Space Explorer app
- [ ] Look for the article citation at the top of the page
- [ ] Click the citation link and confirm it goes to https://doi.org/10.1162/QSS.a.471

**✓ Check 3: Footer Links**
- [ ] Scroll to the bottom of the app page
- [ ] Verify the "Useful Links" section shows the DOI link
- [ ] Click it to confirm it works

**✓ Check 4: View Logs (optional)**
- [ ] Check for any errors: `tail -f /var/log/shiny-server.log`
- [ ] The app should start without import or runtime errors

---

## If Something Goes Wrong

**If the app won't start:**
1. Check the error messages: `sudo journalctl -u shiny-server -n 50`
2. Restore from backup: `cp app.py.backup app.py`
3. Restart: `sudo systemctl restart shiny-server`

**If links don't work:**
1. Double-check the URLs are correct (copy-paste is recommended)
2. Verify no extra spaces or typos were introduced
3. Hard-refresh your browser: Press `Ctrl+Shift+R` (or `Cmd+Shift+R` on Mac)

**If you're still stuck:**
- Contact the development team with:
  - The error message from the logs
  - Which files you updated
  - The exact changes you made

---

## Technical Details (For Reference)

### Files Modified:
1. **app.py** - Main Shiny application file
   - Updated article citation URLs in two locations
   - Old imports of unused modules removed

2. **index.html** - Landing page
   - Complete redesign with modern styling
   - Two main buttons for app and article
   - New copyright year (2025)

3. **README.md** - Project documentation
   - Updated reference to new PDF file location

### What Stayed the Same:
- App functionality remains unchanged
- All interactive features work exactly as before
- Data files are unaffected
- No changes to Python version requirements

---

## Questions or Issues?

If you need help:
1. Check the logs on the server
2. Review this document step-by-step
3. Contact the development team with error messages

Thank you for updating the app! 🎉
