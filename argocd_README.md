
# ArgoCD demo
This guide assumes you have a grounding in the tools that Argo CD is based on. Please read [understanding the basics](https://argo-cd.readthedocs.io/en/stable/understand_the_basics/) to learn about these tools. From [ArgoCD Guidline](https://argo-cd.readthedocs.io/en/stable/getting_started/)
## Prequirements
- Installed [kubectl](https://kubernetes.io/docs/tasks/tools/) command-line tool.
- Have a [kubeconfig](https://kubernetes.io/docs/tasks/access-application-cluster/configure-access-multiple-clusters/) file (default location is ~/.kube/config).
- CoreDNS. Can be enabled for microk8s by microk8s enable dns && microk8s stop && microk8s start

Optional: Use [Minikube](https://kubernetes.io/vi/docs/tasks/tools/install-minikube/) or [Docker Desktop](https://www.docker.com/products/docker-desktop/) to install all there
## 1. Deploy ArgoCD

```
$ kubectl create namespace argocd
$ kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
```
## 2. Download and install ArgoCD CLI
Download the latest Argo CD version from [this repository](https://github.com/argoproj/argo-cd/releases/latest). More detailed installation instructions can be found via the [CLI installation documentation](https://argo-cd.readthedocs.io/en/stable/cli_installation/).
## 3. Access The Argo CD API Server
- Load Balancer
```
$ kubectl patch svc argocd-server -n argocd -p '{"spec": {"type": "LoadBalancer"}}'
```
- Ingress
Follow the [ingress documentation](https://argo-cd.readthedocs.io/en/stable/operator-manual/ingress/) on how to configure Argo CD with ingress.
Example for Ingress
```
# Instal nginx ingress
$ kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# Check ingress nginx
$ kubectl get pods -n ingress-nginx
```
You have to wait until you see:
```
ingress-nginx-controller-xxxxx   Running
```
Apply ingress to ArgoCD
```
$ kubectl apply -f argocd-ingress.yaml
```
Finally, you need to update `/etc/hosts` to point `argocd.example.com` → your ingress IP.
## 4. Port-forward
```
$ kubectl port-forward svc/argocd-server -n argocd 8080:443
```
## 5. Login Using CLI
- Account Initialization: 
The initial `password` for the `admin` account is auto-generated and stored as clear text in the field `password` in a secret named `argocd-initial-admin-secret` in your Argo CD installation namespace. You can simply retrieve this password using the `argocd` CLI:
```
$ argocd admin initial-password -n argocd
```
- Login:
Using the username admin and the password from above, login to Argo CD's IP or hostname:
```
$ argocd login <ARGOCD_SERVER> # http://localhost:8080
```