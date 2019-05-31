import urllib
import re
import http.cookiejar

for i in range(100):
    url = r'http://ucenter.unittec.com/cas/login'
    cjhdr  = urllib.request.HTTPCookieProcessor()
    opener = urllib.request.build_opener(cjhdr)#,proxy)
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.154 Safari/537.36 LBBROWSER')]
    res = opener.open(url)
    content = res.read()
    #从文件中获取信息
    #c0
    #获取form
    m = re.search(rb"name='csrfmiddlewaretoken' value='(\w+?)'",content)
    if m == None:
        print('格式改了，需要调整脚本')

    cok = m.group(1).decode()
    m = re.search(rb'/captcha/image/(\w+?)/',content)
    if m == None:
        print('格式改了，需要调整脚本')
    pngpath = m.group(0).decode()
    captcha_0 = m.group(1).decode()
    #获取图片并显示
    pngurl = urllib.parse.urljoin(url,pngpath)
    res = opener.open(pngurl)
    content = res.read()
    open("pic/cap{0}.png".format(i), "wb").write(content)