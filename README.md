# Text 2 PDF Cover Letter Generator

## Why? 
Applying to jobs in an industry where every job sees hundreds of applicants, makes it a pure numbers game. If one can send out applications faster, then they can send out more applications to better their chances. Websites that offer services to build cover letters hide better features like having multiple cover letters, using different formats and other quality of life features behind paywalls, which is what frustrated me enough to implement one of my own. 


## How to use?  

1.  Ensure you have `my_info.json` with your details.
2.  Run: `python generate_pdf.py --company "Company Name" --body body_file.txt [--manager "Hiring Manager"] [--keywords "keyword1" "keyword2"]`
    *   Replace `"Company Name"` with the company you're applying to.
    *   Replace `body_file.txt` with your cover letter content file in the `content` directory. I normally use `body_<company_name>.txt` for easier management. **Note: Only include the body of the cover letter, opening and closing details are handled in the generation from the template and your info**
    *   Optionally, include `--manager` followed by the hiring manager's name.
    *   Optionally, use `--keywords` followed by a space-separated list of keywords to highlight.

### Using an LLM to Generate Content

If you intend to use this script with an LLM ( i.e. using the llm to generate a cover letter), then you can add the following to your prompt, so that the model will also give you the command to run the script. 
I dont make any comments about the ethicality of this or whether it is correct or not, in the end the script simply converts the text body to the pdf you want it to be. 

**LLM Prompt Template**

Copy and paste this into your LLM, filling in the details. After getting the response, save the cover letter body to the specified `.txt` file in the `content` directory and run the generated command.

```text
Once you have generated the cover letter, provide the exact python script command to generate a PDF from it.

**Command Details:**
- Company: "[Insert Company Name]"
- Body file name: "body_[Insert Company Name without spaces].txt"
- Hiring Manager: "[Insert Manager's Name or leave blank if unknown]"
- Keywords to highlight: "[Insert space-separated keywords from job description]"

The command must be in a bash code block and follow this format: `python generate_pdf.py --company "..." --body "..." [--manager "..."] [--keywords "..." "..." ...]`


```
