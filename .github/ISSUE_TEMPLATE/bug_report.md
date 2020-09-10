---
name: Bug report
about: Please submit issues on github using the following guidelines
title: 'Issue with...'
labels: bug
assignees: ''

---

## Before you submit an issue please check your pymer4 installation!

Problems often arise because pymer4 is installed with incompatible
Python or R packages and can be resolved quickly by
[installing via Anaconda](http://eshinjolly.com/pymer4/installation.html#using-anaconda-recommended).

## Warning: `pip install pymer4` is officially not supported

Installing pymer4 with `pip` is for custom installations by experienced
Python and R programmers. It is not feasible to support these `pip`
installations because pymer4 requires specific versions of Python and
R packages but `pip` only manages Python. If you do install pymer4 with
`pip` and run into trouble, the guidelines for submitting an
issue are largely the same except for one difference: 

*Please try to replicate the issue with the latest version of pymer4 in a conda environment as described next.*

# Issue guidelines

Thank you for helping to improve pymer4!

## Please answer these questions:

Does installing pymer4 into a clean conda environment in any of these
four ways solve the problem? If so, which worked and which didn't?

1. `conda create --name test_env1 -c ejolly -c defaults -c conda-forge pymer4`
2. `conda create --name test_env2 -c ejolly -c conda-forge -c defaults pymer4`
3. `conda create --name test_env3 -c ejolly/label/pre-release -c defaults -c conda-forge pymer4`
4. `conda create --name test_env4 -c ejolly/label/pre-release -c conda-forge -c defaults pymer4`

Explanation of the above commands:

1. Use latest stable release and prefer Anaconda main channel for dependencies, falling back to conda-forge
2. Same as above, but use conda-forge only for all dependencies
3. Same as 1 but use the latest development release
4. Same as 2 but use the latest development release
  

## Please provide the following information

### 1. Description
A clear and concise description of what the problem is, specifically:
- What you expected to happen and what actually happened.
- Anything you tried to solve the issue.

### 2. Minimal reproducible example
These are the shortest steps that reconstruct the problem.
- the command or commands you ran to to install pymer4: `conda install pymer4 ...`
- a Python code snippet, the shorter the better, that runs and exhibits the issue

### 3. Conda environment
Activate the conda environment that has pymer4 installed, run the
following command in a terminal window, and upload the `pymer4_issue.txt` file as
an attachment with your issue.
```
conda list --explicit > pymer4_issue.txt
```

### 4. System information
Please provide the specifics about your computer hardware architecture and
operating system version. For example:

- Linux, in a terminal window

```
	$ uname -mprsv
	Linux 3.10.0-957.21.3.el7.x86_64 #1 SMP Tue Jun 18 16:35:19 UTC 2019 x86_64 x86_64
	
	$ cat /etc/*-release
	CentOS Linux release 7.3.1611 (Core) 
	NAME="CentOS Linux"
	VERSION="7 (Core)"
	ID="centos"
	ID_LIKE="rhel fedora"
	VERSION_ID="7"
	PRETTY_NAME="CentOS Linux 7 (Core)"
	ANSI_COLOR="0;31"
	CPE_NAME="cpe:/o:centos:centos:7"
	HOME_URL="https://www.centos.org/"
	BUG_REPORT_URL="https://bugs.centos.org/"
	
	CENTOS_MANTISBT_PROJECT="CentOS-7"
	CENTOS_MANTISBT_PROJECT_VERSION="7"
	REDHAT_SUPPORT_PRODUCT="centos"
	REDHAT_SUPPORT_PRODUCT_VERSION="7"
	
	CentOS Linux release 7.3.1611 (Core) 
	CentOS Linux release 7.3.1611 (Core) 
```


- Mac OSX, in a terminal window

```
	$ uname -mprsv
	Darwin 19.6.0 Darwin Kernel Version 19.6.0: Thu Jun 18 20:49:00 PDT 2020; root:xnu-6153.141.1~1/RELEASE_X86_64 x86_64 i386

	$ sw_vers
	ProductName:	Mac OS X
	ProductVersion:	10.15.6
	BuildVersion:	19G2021
```

  
- Windows, in a Windows Command Window

```
	C:\Users\some_user> systeminfo

	Host Name:                 DESKTOP-G57OVSM
	OS Name:                   Microsoft Windows 10 Home
	OS Version:                10.0.18362 N/A Build 18362
	OS Manufacturer:           Microsoft Corporation
	OS Configuration:          Standalone Workstation
	OS Build Type:             Multiprocessor Free
	Registered Owner:          some_user
	Registered Organization:
	Product ID:                00326-00840-79774-AAOEM
	Original Install Date:     7/29/2019, 6:13:59 AM
	System Boot Time:          9/9/2020, 9:07:46 AM
	System Manufacturer:       System manufacturer
	System Model:              System Product Name
	System Type:               x64-based PC
	Processor(s):              1 Processor(s) Installed.
    [01]: Intel64 Family 6 Model 158 Stepping 12 GenuineIntel ~3600 Mhz
	BIOS Version:              American Megatrends Inc. 0606, 8/31/2018
	Windows Directory:         C:\WINDOWS
	System Directory:          C:\WINDOWS\system32
	Boot Device:               \Device\HarddiskVolume2
	System Locale:             en-us;English (United States)
	Input Locale:              en-us;English (United States)
	Time Zone:                 (UTC-08:00) Pacific Time (US & Canada)
	Total Physical Memory:     16,305 MB
	Available Physical Memory: 13,598 MB
	Virtual Memory: Max Size:  18,737 MB
	Virtual Memory: Available: 14,542 MB
	Virtual Memory: In Use:    4,195 MB
```
