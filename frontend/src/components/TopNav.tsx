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
import type { SVGProps } from "react";

function GithubIcon(props: SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true" {...props}>
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.3 3.438 9.8 8.207 11.387.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.725-4.042-1.61-4.042-1.61-.546-1.386-1.332-1.755-1.332-1.755-1.089-.744.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.809 1.304 3.495.997.108-.775.418-1.305.762-1.605-2.665-.303-5.467-1.332-5.467-5.93 0-1.31.469-2.381 1.236-3.221-.124-.303-.536-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.29-1.552 3.297-1.23 3.297-1.23.655 1.652.243 2.873.12 3.176.77.84 1.235 1.911 1.235 3.221 0 4.61-2.807 5.624-5.48 5.921.43.372.823 1.102.823 2.222 0 1.604-.014 2.898-.014 3.293 0 .322.216.696.825.578C20.565 21.796 24 17.298 24 12c0-6.63-5.373-12-12-12Z" />
    </svg>
  );
}

const centerLinks = [
  { label: "Trace", href: "#" },
  { label: "Activation patching", comingSoon: true },
  { label: "Attention heatmaps", comingSoon: true },
  { label: "Logit attribution", comingSoon: true },
];

const rightLinks = [
  { label: "Docs", href: "#" },
  {
    label: "GitHub",
    href: "https://github.com/blaketylerfullerton/mechlens",
    icon: GithubIcon,
  },
];

function NavLinks({ links }: { links: typeof centerLinks | typeof rightLinks }) {
  return (
    <NavigationMenuList>
      {links.map((link) =>
        "comingSoon" in link && link.comingSoon ? (
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
            <NavigationMenuLink
              href={link.href}
              aria-label={"icon" in link && link.icon ? link.label : undefined}
              className={navigationMenuTriggerStyle()}
            >
              {"icon" in link && link.icon ? <link.icon className="size-4" /> : link.label}
            </NavigationMenuLink>
          </NavigationMenuItem>
        )
      )}
    </NavigationMenuList>
  );
}

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
        <NavLinks links={centerLinks} />
      </NavigationMenu>
      <NavigationMenu className="justify-self-end">
        <NavLinks links={rightLinks} />
      </NavigationMenu>
    </header>
  );
}
