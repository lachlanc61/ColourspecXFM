from xfmreadout import diagops as diagops

args = ["-f", "/mnt/d/DATA/XFMDATA/2023/Lachlan/REFERENCE/uMatter/Mo_vac_230313/50-200_TC1p0/diagnostics_um200.log",]

#args = ["-f", "/mnt/d/DATA/XFMDATA/2023/Lachlan/REFERENCE/uMatter/Mo_vac_230313/50-200_TC1p0/diagnostics.log", "-s"]

diagops.main(args)