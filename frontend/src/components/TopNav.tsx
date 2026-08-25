import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu";
import { VariableIcon } from '@heroicons/react/24/solid'
const links = [
  { label: "Trace", href: "#" },
  { label: "Docs", href: "#" },
  { label: "GitHub", href: "https://github.com/blaketylerfullerton/mechlens" },
  { label: "Activation patching", comingSoon: true },
  { label: "Attention heatmaps", comingSoon: true },
  { label: "Logit attribution", comingSoon: true },
];

export function TopNav() {
  return (
    <header className="grid grid-cols-3 items-center border-b border-border px-8 py-4">
      <span className="flex items-center">
        <span className="inline-flex items-center text-lg font-semibold text-foreground">
          M
          <VariableIcon className="size-5" />
          chlens
        </span>
      </span>
      <NavigationMenu className="justify-self-center">
        <NavigationMenuList>
          {links.map((link) =>
     
            link.comingSoon ? (
              <NavigationMenuItem key={link.label}>
                <span
                  className={cn(
                    navigationMenuTriggerStyle(),
                    "cursor-not-allowed gap-1.5 text-muted-foreground"
                  )}
                  aria-disabled="true"
                >
                  {link.label}
                  <Badge variant="outline" className="text-[10px]">
                    Soon
                  </Badge>
                </span>
              </NavigationMenuItem>
            ) : (
              <NavigationMenuItem key={link.label}>
                <NavigationMenuLink href={link.href} className={navigationMenuTriggerStyle()}>
                  {link.label}
                </NavigationMenuLink>
              </NavigationMenuItem>
            )
          )}
        </NavigationMenuList>
      </NavigationMenu>
      <div />
    </header>
  );
}
