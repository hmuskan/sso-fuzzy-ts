from rouge import FilesRouge
files_rouge = FilesRouge()
scores = files_rouge.get_scores('6.summarized/001.txt', '3.reference/001.txt', avg=True)
print(scores)
# or
#scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
