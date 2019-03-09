# The Visual Studio Code configure git to GitHub
1、Download and install git, and then enter the command:
    >>> git config user.name 'your_name' -g (your_name means the github username)
    >>> git config user.email 'your_email' -g (your_email means the github email)

2、Create a new repository in GitHub
    Get the repository SSH address

3、Create a directory locally
    Right click on the directory and select 'git Bash here', run the following code in order on the Git Bash.
    >>> echo "# vscode-demo" >> README.md
    >>> git init
    >>> git add README.md
    >>> git commit -m "first commit"
    >>> git remote add origin git@github.com:***/***.git
    >>> git push -u origin master

    if is error and the message is "fatal:Could not read from remote repository":
    enter the command on the Git Bash:
    >>> ssh-keygen -t rsa -b 4096 -C "your_email" (your_email means the github email)
    Press three times to enter, and then get the SSH key by the command:
    >>> cat ~/.ssh/id_rsa.pub
    Next, Click on the personal icon in the top right corner of the GitHub Homepage,
    Settings --> SSH and GPG keys --> New SSH key title --> New SSH key,
    Enter the SSH address generate in the Git Bash into the input box.
    Rerun the code in order on the Git Bash.
     