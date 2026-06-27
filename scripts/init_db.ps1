# 在 Windows 上创建 PostgreSQL 数据库 hr_resume
# 用法：在项目根目录执行
#   .\scripts\init_db.ps1
# 或指定密码：  .\scripts\init_db.ps1 -Password 20020312gjx

param([string]$Password = $env:PGPASSWORD)
if (-not $Password) { $Password = Read-Host "输入 postgres 密码" }
$env:PGPASSWORD = $Password
psql -U postgres -h localhost -c "CREATE DATABASE hr_resume;" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "若数据库已存在可忽略。否则检查：PostgreSQL 已启动、密码正确、psql 在 PATH 中。" }
