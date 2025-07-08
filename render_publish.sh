#!/bin/bash
set -e  # Exit on any error
hugo
echo "Building site with Hugo..."
echo "Running pagefind..."
xsltproc --output public/sitemap.xml filter_sitemap.xsl public/sitemap.xml
git add .
current_date=$(date +"%Y-%m-%d %H-%M-%S")
commit_message=${1:-"Release $current_date"}
git commit -m "$commit_message"
git push
echo "Deployment completed successfully!"