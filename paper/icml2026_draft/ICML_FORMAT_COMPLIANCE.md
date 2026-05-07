# ICML 2026 Format Compliance Checklist

Sources:

- ICML 2026 formatting instructions PDF text dump provided in chat.
- Mechanistic Interpretability Workshop at ICML 2026 CFP: <https://mechinterpworkshop.com/cfp/>
- ICML 2026 Author Instructions: <https://icml.cc/Conferences/2026/AuthorInstructions>
- ICML 2026 Code of Conduct: <https://icml.cc/Conferences/2026/CodeOfConduct>
- ICML 2026 general Call for Papers: <https://icml.cc/Conferences/2026/CallForPapers>

This checklist is intentionally syntactic and submission-focused. It complements
`REVIEW_RUBRIC.md`, which covers scientific framing, evidence quality, and review
questions.

## Source Priority For This Workshop Submission

- [ ] Treat the Mechanistic Interpretability Workshop CFP as the controlling source for workshop-specific submission rules.
- [ ] Treat ICML 2026 Author Instructions and the ICML template/example paper as controlling for ICML-format syntax and typesetting.
- [ ] Treat the general ICML CFP as background policy unless the workshop CFP or workshop OpenReview site explicitly imports a rule.
- [ ] Resolve any conflicts against the live workshop CFP/OpenReview instructions before final submission.
- [ ] Note current file-size discrepancy: the formatting PDF says total file size may not exceed 10 MB, while live ICML Author Instructions say submission PDF max is 50 MB and camera-ready max is 20 MB.
- [ ] Until workshop OpenReview confirms otherwise, keep the PDF well below 10 MB if feasible.

## Mechanistic Interpretability Workshop CFP

- [ ] Submit through OpenReview.
- [ ] Submission deadline is May 8, 2026 AoE.
- [ ] Authors will be notified of acceptance by June 12, 2026 AoE.
- [ ] The workshop is non-archival.
- [ ] The reviewing process is double-blind.
- [ ] Authors are responsible for ensuring no identifying details are included.
- [ ] Search the manuscript for names of all core contributors before submission.
- [ ] Search the manuscript for GitHub usernames of all core contributors before submission.
- [ ] Search the manuscript for HuggingFace usernames of all core contributors before submission.
- [ ] All submissions must be made via OpenReview.
- [ ] If an author lacks an institutional email, account approval can take up to two weeks; plan OpenReview access accordingly.
- [ ] At least one reciprocal reviewer is required per submission.
- [ ] Each reciprocal reviewer will be assigned 3 papers to review.
- [ ] A person can be reciprocal reviewer on at most 3 workshop papers.
- [ ] Initial workshop submissions may use either ICML 2026 format or NeurIPS 2026 format.
- [ ] Since this draft uses ICML format, enforce ICML workshop page limits.
- [ ] ICML-format short papers have a maximum of 4 pages.
- [ ] ICML-format long papers have a maximum of 8 pages.
- [ ] NeurIPS-format short papers have a maximum of 5 pages.
- [ ] NeurIPS-format long papers have a maximum of 9 pages.
- [ ] Page limits exclude references.
- [ ] Page limits exclude appendices.
- [ ] References and appendices are unlimited.
- [ ] Reviewers are not expected to read appendices.
- [ ] Accepted papers must be converted to ICML format for camera-ready.
- [ ] Accepted papers get one additional camera-ready page to integrate reviewer feedback.
- [ ] Long works are held to a higher standard of rigor and depth than short works.
- [ ] Authors are encouraged but not required to attend the workshop in person.
- [ ] Authors are strongly encouraged to open source code.
- [ ] Authors are strongly encouraged to open source models.
- [ ] Authors are strongly encouraged to open source prompts.
- [ ] Authors are strongly encouraged to open source data.
- [ ] Authors are strongly encouraged to open source interactive demos.
- [ ] Reviewers will be asked to consider reproducibility, code access, and/or data access.
- [ ] Use an anonymous GitHub-sharing service, anonymous repository, or equivalent if sharing code during review.
- [ ] For larger files such as model weights or datasets, use an anonymous HuggingFace account or equivalent anonymous hosting.
- [ ] Interactive demos should be hosted anonymously during review.
- [ ] Work already accepted to ICML 2026 is welcome at the workshop.
- [ ] If fast-tracking an ICML-accepted paper, provide evidence of previous reviews and acceptance.
- [ ] Fast-track submissions are still subject to additional reviews, such as theme fit.
- [ ] Do not submit work accepted to an archival venue other than ICML 2026.
- [ ] Papers previously accepted to the 2025 Mech Interp Workshop, including extended or expanded versions, will not be accepted.
- [ ] Papers previously rejected from the 2025 Mech Interp Workshop must be meaningfully revised or they will be desk-rejected.
- [ ] Submissions undergoing peer review elsewhere at the workshop deadline are welcome.
- [ ] The paper should convincingly argue how it furthers mechanistic interpretability.
- [ ] Strong empirical work should clearly state specific falsifiable hypotheses and explain how evidence supports or does not support them.
- [ ] Alternatively, strong empirical work should convincingly show practical benefits over well-implemented baselines.
- [ ] Document strengths and weaknesses of the evidence.
- [ ] Do not downplay significant limitations.
- [ ] Do not omit significant limitations.

## Workshop Fit Checklist

- [ ] The work should further the ability to use neural-network internal states to understand neural networks.
- [ ] The paper should fit at least one workshop topic of interest.
- [ ] If framed as understanding model internals, explain what is learned about representations, feature geometry, or internal use.
- [ ] If framed as mechanistic discovery, explain how the work uncovers or validates internal structure.
- [ ] If framed as practical interpretability, explain the downstream application, development, debugging, or evaluation relevance.
- [ ] If framed as safety/monitoring/model repair, explain the safety-relevant use.
- [ ] If framed as scaling/generalization/automation, explain how insights generalize beyond controlled settings or across model scale.
- [ ] If framed as conceptual/foundational work, clarify the framework or definitions advanced.
- [ ] Rigorous negative results are in scope.
- [ ] Rigorous replications are in scope.
- [ ] Critiques and compelling failed replications are in scope.
- [ ] Open-source tools, models, datasets, educational materials, distillations, and position pieces are in scope when relevant.

## Submission Package

- [ ] Submit the paper as a PDF, not Word or any other format.
- [ ] Use LaTeX; live ICML Author Instructions state there is no support for typesetting software other than LaTeX.
- [ ] Initial submission must be anonymous and double-blind.
- [ ] The submitted PDF must include main body, references, and appendices in one file.
- [ ] Do not submit appendices as a separate PDF; reviewers may miss separate appendix files.
- [ ] Main body must fit within 8 pages for initial submission, excluding references and appendices.
- [ ] Final camera-ready version may add one extra main-body page.
- [ ] References and appendices have no page limit.
- [ ] Total PDF file size must not exceed 10 MB.
- [ ] Live ICML Author Instructions list a 50 MB maximum for submission PDF and 20 MB for camera-ready; confirm workshop OpenReview limit before final upload.
- [ ] Reviewers are not required to look beyond the first 8 pages of the submitted document.
- [ ] Reviewers are not required to inspect supplementary material.
- [ ] Material critical to evaluating the paper must be in the main body, not only appendix or supplementary material.
- [ ] All supplementary material must be submitted by the same deadline as the paper submission.

## Anonymous Submission

- [ ] Use anonymous ICML style for review: `\usepackage{icml2026}`, not `\usepackage[accepted]{icml2026}`.
- [ ] Do not include author names on the title page.
- [ ] Do not include author affiliations on the title page.
- [ ] Do not include identifying author information anywhere in the paper.
- [ ] Do not include acknowledgements in the initial anonymous submission.
- [ ] Do not include grant numbers in the initial anonymous submission.
- [ ] Do not include institution-identifying URLs in the initial submission.
- [ ] Do not include links to public, non-anonymized code repositories in the initial submission.
- [ ] If software/data URLs are included for review, they must be anonymous.
- [ ] Public or identity-revealing code/data URLs should wait until camera-ready.
- [ ] Cite published self-work in the third person.
- [ ] Do not write identity-revealing self-citation phrasing such as "in previous work, we showed..."
- [ ] Do not anonymize published papers in the references.
- [ ] Unpublished or under-submission self-cited manuscripts must be anonymized.
- [ ] Anonymized unpublished manuscripts, if cited, must be submitted as Supplementary Material through OpenReview.
- [ ] The paper must remain self-contained even if supplementary material exists.
- [ ] Previously published overlapping author work must be cited in a way that preserves anonymity.
- [ ] Differences from earlier overlapping papers must be explained in the submission text.
- [ ] Papers that explicitly reveal author identity will be rejected.
- [ ] Papers that implicitly reveal author identity will be rejected.
- [ ] If a non-anonymized version is posted online before decisions, the submitted version must not refer to it.
- [ ] Do not advertise the work as an ICML submission during the review period.

## Camera-Ready Differences

- [ ] Camera-ready papers must include author names and affiliations.
- [ ] For accepted papers, switch to `\usepackage[accepted]{icml2026}`.
- [ ] The first-page footnote must change from the under-review wording to the proceedings/copyright wording.
- [ ] Camera-ready pages after the first must have a running head.
- [ ] The running head should be the paper title unless it is too long.
- [ ] If the title is too long for the running head, set a shorter running title with `\icmltitlerunning{...}`.
- [ ] Camera-ready author names should start 0.3 inches below the bottom title rule.
- [ ] Camera-ready author names should be 10 point bold, centered, and separated by whitespace.
- [ ] Author names should not be broken across lines.
- [ ] Use unbolded superscript numbers starting from 1 for affiliations.
- [ ] Affiliations should be numbered in order of appearance.
- [ ] Each distinct affiliation should appear once.
- [ ] Academic affiliations should include Department, University, City, State/Region, Country.
- [ ] Industrial affiliations should use the analogous organization/location format.
- [ ] Multiple affiliations for one author should use multiple superscripts separated by thin spaces.
- [ ] Equal-contribution first authors should have superscript asterisks.
- [ ] If equal contribution is used, include `*Equal contribution` in the footnote block before affiliations.
- [ ] Corresponding authors may be listed after affiliations.
- [ ] Corresponding author emails should use `Full Name <email@domain.com>` format.
- [ ] Ideally list only one or two corresponding authors.
- [ ] Camera-ready version may include acknowledgements.
- [ ] Camera-ready acknowledgements should be an unnumbered section near the end.
- [ ] Camera-ready acknowledgements do not count toward the main page limit.
- [ ] Camera-ready may include software/data URLs when appropriate.
- [ ] Accepted papers' originally submitted manuscript and supplementary material will become public on OpenReview.
- [ ] Accepted papers' anonymized reviews, meta-reviews, rebuttal, and reviewer-author discussion will become public on OpenReview.
- [ ] Camera-ready changes may incorporate reviewer feedback and other improvements only if the essential content remains unchanged from what reviewers saw.
- [ ] Post-conference revisions are allowed but must be made by the post-conference revision deadline.
- [ ] Accepted papers must submit a lay summary/plain-language summary in OpenReview at camera-ready time.

## Style File And Fonts

- [ ] Do not alter `icml2026.sty`.
- [ ] Do not compress the format by reducing vertical spaces.
- [ ] Do not modify margins or spacing to fit more content.
- [ ] Paper should use 10 point Times font throughout body text.
- [ ] Body text should have 11 point vertical spacing.
- [ ] PDF must use embedded Type-1 fonts only.
- [ ] Avoid Type-3 fonts, including those introduced by graphics.
- [ ] Prefer `pdflatex` where possible because it avoids common Type-3 font issues.
- [ ] If `hyperref` causes problems, use the `nohyperref` option with the ICML package.

## Page Geometry

- [ ] Paper must be formatted in two columns.
- [ ] Overall text width should be 6.75 inches.
- [ ] Text height should be 9.0 inches.
- [ ] Column separation should be 0.25 inches.
- [ ] Left margin should be 0.75 inches.
- [ ] Top margin should be 1.0 inch.
- [ ] Final versions must be produced for US letter size.
- [ ] Do not write anything in the margins.

## Title

- [ ] Title should be 14 point bold.
- [ ] Title should be centered between two 1 point horizontal rules.
- [ ] Top title rule should be 1.0 inch below the top edge of the page.
- [ ] Capitalize the first letter of content words in the title.
- [ ] Use lower case for the rest of each title word.
- [ ] Use TeX math in the title only sparingly.
- [ ] Do not use custom macros in the title.
- [ ] Do not use images or other TeX commands in the title.
- [ ] Enter accents and special characters using TeX commands, not raw non-English characters.

## Abstract

- [ ] Abstract must be one paragraph.
- [ ] Abstract should be brief and self-contained.
- [ ] Abstract should be roughly 4-6 sentences.
- [ ] Abstract heading should be centered, bold, and 11 point.
- [ ] Abstract body should be 10 point with 11 point vertical spacing.
- [ ] Abstract body should be indented 0.25 inches more than normal on both left and right margins.
- [ ] Insert 0.4 inches of blank space after the abstract body.

## Sectioning And Paragraphs

- [ ] Use no more than three levels of headings.
- [ ] Section headings should be numbered.
- [ ] Section headings should be flush left.
- [ ] Section headings should be 11 point bold.
- [ ] Section headings should have content words capitalized.
- [ ] Leave 0.25 inches before section headings.
- [ ] Leave 0.15 inches after section headings.
- [ ] Subsection headings should be numbered.
- [ ] Subsection headings should be flush left.
- [ ] Subsection headings should be 10 point bold.
- [ ] Subsection headings should have content words capitalized.
- [ ] Leave 0.2 inches before subsection headings.
- [ ] Leave 0.13 inches after subsection headings.
- [ ] Subsubsection headings should be numbered.
- [ ] Subsubsection headings should be flush left.
- [ ] Subsubsection headings should be 10 point small caps.
- [ ] Subsubsection headings should have content words capitalized.
- [ ] Leave 0.18 inches before subsubsection headings.
- [ ] Leave 0.1 inches after subsubsection headings.
- [ ] Do not indent the first line of a paragraph.
- [ ] Insert a blank line between succeeding paragraphs.

## Footnotes

- [ ] Footnotes should be numbered in the text.
- [ ] Footnotes should appear at the bottom of the column where they are cited.
- [ ] Footnotes should be 9 point.
- [ ] Footnotes should be complete sentences.
- [ ] Precede the first footnote in a column with a 0.8 inch horizontal rule.
- [ ] Multiple footnotes in a column should appear in citation order.
- [ ] Spread multiple footnotes across columns/pages where possible.

## Figures

- [ ] Figures should be centered.
- [ ] Figures should be legible.
- [ ] Figures should be separated from the text.
- [ ] Figure lines should be dark.
- [ ] Figure lines should be at least 0.5 points thick.
- [ ] Text should not appear on a gray background.
- [ ] Label all distinct components of each figure.
- [ ] Graph axes must be named.
- [ ] Graphs should include legends briefly describing curves.
- [ ] Do not include a title inside the figure graphic.
- [ ] Use the caption instead of an in-graphic title.
- [ ] Number figures sequentially.
- [ ] Place figure number and caption below the graphic.
- [ ] Leave at least 0.1 inches before the figure caption.
- [ ] Leave at least 0.1 inches after the figure caption.
- [ ] Figure captions should be 9 point.
- [ ] Center one-line figure captions.
- [ ] Set figure captions flush left when they run two or more lines.
- [ ] Figures may float to the top or bottom of a column.
- [ ] Wide figures may span both columns using `figure*`.
- [ ] Two-column figures must be placed at the top or bottom of the page.

## Graphics Files

- [ ] Graphics files should be reasonably sized.
- [ ] Use vector formats such as `.eps` or `.pdf` for plots when possible.
- [ ] Use lossless bitmap formats such as `.png` for raster graphics with sharp lines.
- [ ] Use JPEG only for photo-like images.
- [ ] Check imported graphics for Type-3 font problems.

## Algorithms

- [ ] Use the LaTeX `algorithm` environment for pseudocode.
- [ ] Use the LaTeX `algorithmic` environment for pseudocode internals.
- [ ] Include `algorithm.sty` and `algorithmic.sty` as needed.

## Tables

- [ ] Tables should be centered.
- [ ] Tables should be legible.
- [ ] Tables should be numbered consecutively.
- [ ] Place table titles/captions above the table.
- [ ] Leave at least 0.1 inches before the table title.
- [ ] Leave at least 0.1 inches after the table title.
- [ ] Table titles should be 9 point.
- [ ] Center one-line table titles.
- [ ] Set table titles flush left when they run two or more lines.
- [ ] Tables should contain textual material, not graphical material.
- [ ] Specify contents of each row and column in the topmost row.
- [ ] Tables may float to the top or bottom of a column.
- [ ] Wide tables may span both columns.
- [ ] Two-column tables must be placed at the top or bottom of the page.

## Theorems And Formal Statements

- [ ] Theorems, definitions, propositions, lemmas, corollaries, assumptions, and remarks should be numbered consecutively within sections.
- [ ] Formal statement numbering should follow the section number, e.g. Definition 2.1.
- [ ] Proofs should use the standard proof formatting supplied by the style/template.

## Citations

- [ ] Use APA reference format.
- [ ] Use `natbib.sty` and `icml2026.bst` when relying on LaTeX bibliography tooling.
- [ ] In-text citations should include authors' last names and year.
- [ ] If author names appear in the sentence, put only the year in parentheses.
- [ ] Otherwise, put author names and year together in parentheses.
- [ ] Separate multiple references with semicolons.
- [ ] Place multiple citations in chronological order.
- [ ] Use `et al.` only for citations with three or more authors, or after all authors were listed earlier.
- [ ] Self-citations in anonymous submissions must be written in the third person.

## References

- [ ] Use an unnumbered first-level section heading for references.
- [ ] References should use hanging indent style.
- [ ] First line of each reference should be flush with the left margin.
- [ ] Subsequent reference lines should be indented by 10 points.
- [ ] Alphabetize references by first-author surname.
- [ ] Single-author entries should precede multiple-author entries for the same first author.
- [ ] References by the same authors should be ordered by year, earliest first.
- [ ] References must be as complete as possible.
- [ ] Include page numbers whenever possible.
- [ ] Use current/actual author names.
- [ ] Keep reference formatting presentable and consistent.
- [ ] Protect capital letters of names and abbreviations in BibTeX titles, e.g. `{B}ayesian`, `{L}ipschitz`.

## Impact Statement

- [ ] Include an Impact Statement.
- [ ] Impact Statement should discuss potential broader impact, including ethical aspects and future societal consequences.
- [ ] Impact Statement must be an unnumbered section.
- [ ] Impact Statement should appear at the end of the paper.
- [ ] Impact Statement must appear before References.
- [ ] If Acknowledgements are present in camera-ready, Impact Statement and Acknowledgements may appear in either order.
- [ ] Impact Statement does not count toward the main paper page limit.
- [ ] A short generic statement is allowed when no specific societal consequences need to be highlighted.

## Acknowledgements

- [ ] Do not include acknowledgements in the initial anonymous submission.
- [ ] Camera-ready acknowledgements may be included.
- [ ] Acknowledgements should be placed near the end in an unnumbered section.
- [ ] Acknowledgements should appear before References if present.
- [ ] Acknowledgements do not count toward the main paper page limit.

## Accessibility

- [ ] Make the submission as accessible as possible for readers with disabilities.
- [ ] Consider readers with sensory or neurological differences.
- [ ] Follow any additional accessibility guidance posted on the ICML website.

## Software And Data

- [ ] If accepted, publish software and data with the camera-ready version when appropriate.
- [ ] Camera-ready software/data links may be included as URLs.
- [ ] Initial submissions must not include URLs revealing institution or identity.
- [ ] For review, use anonymous URLs or upload material as Supplementary Material in OpenReview.
- [ ] Do not rely on reviewers inspecting software, data, or supplementary material.
- [ ] Supplementary material can be a supplementary manuscript.
- [ ] Supplementary material can be code/data.
- [ ] If the paper makes an anonymous reference, upload the referenced paper as supplementary material.
- [ ] Supplementary material must also be anonymized.
- [ ] Traditional text appendices do not need separate supplementary upload when they are included in the main PDF.
- [ ] Supplementary code may be submitted as a zip file.
- [ ] Supplementary code may be submitted as a PDF.
- [ ] Anonymize submitted code by removing author names.
- [ ] Anonymize submitted code by removing licenses if they identify authors.
- [ ] Anonymous GitHub repositories are allowed for code.
- [ ] Anonymous GitHub repositories must be on a branch that will not be modified after the submission deadline.
- [ ] If using a GitHub link for supplementary code, put the link in a standalone text file inside the submitted zip.
- [ ] Data submissions are welcome only if the authors have the right to share the data.
- [ ] If final paper refers to supplementary code/data, authors must provide an archival link to a suitable repository.

## Appendix

- [ ] Appendix may contain unlimited text, subject to the total PDF size limit.
- [ ] Appendix should be included in the same PDF as the main body and references.
- [ ] Main body must remain at most 8 pages for initial submission regardless of appendix length.
- [ ] Camera-ready main body may be 9 pages.
- [ ] The appendix may be one-column if `\onecolumn` is kept.
- [ ] The appendix may be two-column if `\onecolumn` is removed.
- [ ] Apart from optional one-column vs two-column appendix layout, the style must remain unchanged.
- [ ] Appendix must keep the same font size, spacing, margins, page numbering, and general style as the main body.

## Author Response And Review Period

- [ ] Do not expect the submitted paper to be publicly accessible during review.
- [ ] Author responses must preserve double-blind anonymity.
- [ ] Author responses must not contain identity-revealing information.
- [ ] Author responses must not contain non-anonymized URLs.
- [ ] Author responses must not contain personal-website URLs.
- [ ] Author responses must not contain shortened URLs such as TinyURL links.
- [ ] Reviewers are not expected to follow external URLs in author responses.
- [ ] There is no option to upload a revised paper during the author feedback period.
- [ ] Author responses should use professional and polite language.

## General ICML CFP Policies

- [ ] Main-conference deadlines are strict and receive no extensions.
- [ ] All authors must have OpenReview accounts.
- [ ] OpenReview institutional-email registration is strongly recommended.
- [ ] OpenReview accounts without institutional email can take up to two weeks to activate.
- [ ] For main-conference submissions, title, complete author list, and complete abstract must be in the submission form by the abstract deadline.
- [ ] For main-conference submissions, placeholder abstracts that are substantially rewritten by full submission risk removal without consideration.
- [ ] For main-conference submissions, author list cannot be changed after the abstract deadline without written justification and case-by-case program-chair approval.
- [ ] General ICML requires at least one qualified reciprocal reviewer per submission unless an exception applies.
- [ ] General ICML caps one author as reciprocal reviewer for at most 2 of that author's main-conference submissions.
- [ ] General ICML may desk-reject submissions that fail reciprocal-review requirements.
- [ ] Workshop reciprocal-review rules differ: use the workshop-specific rule for this workshop submission.
- [ ] Authors may post versions of their work on preprint servers.
- [ ] Authors may give talks on submitted work during review.
- [ ] Authors must not advertise the work as an ICML submission during review.
- [ ] Do not submit identical or substantially similar work already published, accepted, or concurrently submitted to another archival conference or journal.
- [ ] Workshop papers without published proceedings do not violate the general ICML dual-submission policy.
- [ ] Concurrent ICML submissions with overlapping authors are treated as prior work.
- [ ] All claims should be clearly stated.
- [ ] All claims should be supported by reproducible experiments and/or sound theoretical analysis.
- [ ] Contributions should be situated in broader scientific and ML literature.
- [ ] Relevant prior work should be acknowledged and differentiated.

## Generative AI And Integrity

- [ ] Authors may use generative AI tools, including LLMs, for writing or research assistance.
- [ ] Authors remain fully responsible for all paper content.
- [ ] Authors are responsible for avoiding AI-generated plagiarism.
- [ ] Authors are responsible for avoiding AI-generated scientific misconduct.
- [ ] Authors are encouraged to explain notable ways generative AI tools were used in the research methodology.
- [ ] LLMs are not eligible for authorship.
- [ ] Prompt injection is strictly forbidden.
- [ ] Prompt injection attempts will result in desk rejection.
- [ ] Plagiarism in any form is forbidden.
- [ ] Advertising the work as under submission to ICML during review is forbidden.
- [ ] Collusion with reviewers, ACs, or SACs is forbidden.
- [ ] Suspected ethics violations may be reported to ICML.
- [ ] Submissions may be rejected or sanctioned for ethical violations.

## Code Of Conduct

- [ ] All participants must agree to the ICML Code of Conduct.
- [ ] Code of Conduct applies to attendees, organizers, reviewers, speakers, sponsors, and volunteers.
- [ ] Code of Conduct applies to conference sessions, workshops, conference-sponsored social events, and official communication channels including social media.
- [ ] Maintain respectful scientific debate.
- [ ] Do not engage in harassment.
- [ ] Do not engage in bullying.
- [ ] Do not engage in discrimination.
- [ ] Do not engage in retaliation.
- [ ] Do not make offensive comments related to gender, gender identity/expression, age, sexual orientation, disability, physical appearance, body size, race, ethnicity, religion, politics, technology choices, or other personal characteristics.
- [ ] Do not engage in intimidation, personal attacks, or sustained disruption.
- [ ] Do not interfere with another participant's full participation.
- [ ] Do not engage in sexual harassment, stalking, following, harassing photography/recording, inappropriate physical contact, unwelcome sexual attention, public vulgar exchanges, or diminutive characterizations.
- [ ] Comply immediately if asked by any community member to stop prohibited behavior.
- [ ] "Just joking" is not an acceptable defense for prohibited behavior.
- [ ] Reports can be made to ICML Inclusion & Accessibility co-chairs or the Conference HR Liaison.
- [ ] Reports during the conference should receive response in less than 24 hours.
- [ ] Reports at other times should receive response in less than two weeks.
