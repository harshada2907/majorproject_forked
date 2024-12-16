from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, logout, authenticate

def uhome(request):
    if request.user.is_authenticated:
        return render(request, "home.html")
    else:
        return redirect("ulogin")

def usignup(request):
    if request.user.is_authenticated:
        return redirect("uhome")
    elif request.method == "POST":
        un = request.POST.get("un")
        pw1 = request.POST.get("pw1")
        pw2 = request.POST.get("pw2")
        if pw1 == pw2:
            try:
                usr = User.objects.get(username=un)
                msg = un + " already registered"
                return render(request, "signup.html", {"msg": msg})

            except User.DoesNotExist:
                usr = User.objects.create_user(username=un, password=pw1)
                usr.save()
                return redirect("ulogin")
        else:
            msg = "password did not match"
            return render(request, "signup.html", {"msg": msg})
    else:
        return render(request, "signup.html")

def ulogin(request):
    if request.user.is_authenticated:
        return redirect("uhome")
    elif request.method == "POST":
        un = request.POST.get("un")
        pw = request.POST.get("pw")
        usr = authenticate(username=un, password=pw)
        if usr is not None:
            login(request, usr)
            return redirect("uhome")
        else:
            msg = "invalid username or password"
            return render(request, "login.html", {"msg": msg}) 
    else:
        return render(request, "login.html")

def ulogout(request):
    logout(request)
    return redirect("ulogin")
